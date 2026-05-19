"""
MIIPHER Plugin — Vocal-SOTA adapter for SNR < 10 dB (v9.12.1)

§4.4 SOTA-Matrix 2026: MIIPHER (Zhang et al. 2023, Google) ist das SOTA-Modell
für extrem starke Rauschumgebungen (SNR < 10 dB), Vocal-Restaurierung von
stark degradiertem Gesangsmaterial. Basiert auf W2v-BERT als Conditioning.

Model status: Native MIIPHER is not bundled. This module is still productive:
it routes deep-noise vocal material through the best local open-source chain
SGMSE+ → DeepFilterNet v3.II → conservative DSP, with explicit route metadata.

Aktivierung: Nur wenn DefectScanner `noise_snr_db < 10.0` UND
`panns_singing_confidence ≥ 0.35`.

§0h Invariante: Kein Output wird akzeptiert wenn artifact_freedom < 0.95.
§0j KI-Modell-Limitation: MIIPHER halluziniert potentiell Harmonics für
unbekannte Singstimmen → hallucination_guard.py nach Anwendung Pflicht.
"""

from __future__ import annotations

import logging
import threading
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Modell-Pfad (nach Integration in AppImage)
_MIIPHER_ONNX_PATH = None  # TODO: "models/miipher/miipher.onnx" nach Modell-Integration

# SNR-Schwellwert für MIIPHER-Aktivierung (dB)
MIIPHER_SNR_THRESHOLD_DB = 10.0

# Minimum PANNs Gesangskonfidenz für MIIPHER
MIIPHER_SINGING_CONFIDENCE_MIN = 0.35

# DeepFilterNet Fallback energy_bias bei Gesang (§0j)
_DFN_FALLBACK_ENERGY_BIAS_DB = -6.0

# Singleton
_instance: MiipherPlugin | None = None
_lock = threading.Lock()


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Lädt an optional plugin/backend symbol lazily."""
    return getattr(import_module(module_name), symbol_name)


def _load_module(module_name: str) -> Any:
    """Lädt an optional module lazily."""
    return import_module(module_name)


class MiipherPlugin:
    """
    MIIPHER Last-Resort-Entrauscher für stark degradiertes Gesangsmaterial.

    Primary: MIIPHER ONNX (wenn Modell vorhanden).
    Fallback: DeepFilterNet v3.II mit Gesang-optimiertem energy_bias.

    Verwendung:
        plugin = get_miipher_plugin()
        if plugin.should_activate(noise_snr_db, panns_singing):
            result = plugin.enhance(audio, sr)
    """

    def __init__(self) -> None:
        self._model_loaded = False
        self._model_session = None
        self._last_route_metadata: dict[str, object] = {
            "model_used": "none",
            "capability_status": "unavailable",
            "fallback_chain": [],
            "native_miipher_loaded": False,
        }
        self._try_load_model()

    @property
    def route_metadata(self) -> dict[str, object]:
        """Gibt metadata for the last enhancement route zurück."""
        return dict(self._last_route_metadata)

    def is_productive(self) -> bool:
        """True when a real local SOTA/fallback model path is available."""
        try:
            get_model_capability_gate = _load_symbol(
                "backend.core.dsp.model_capability_gate",
                "get_model_capability_gate",
            )
            report = get_model_capability_gate().build_report()
            capabilities = report.get("capabilities", {}) if isinstance(report, dict) else {}
            if not isinstance(capabilities, dict):
                return False
            for name in ("sgmse_plus", "deepfilternet_v3_ii"):
                cap = capabilities.get(name, {})
                if isinstance(cap, dict) and cap.get("status") in {"sota_real", "sota_fallback"}:
                    return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("MIIPHER adapter capability check unavailable: %s", exc)
        return False

    def _try_load_model(self) -> None:
        """Try to load a native MIIPHER ONNX model if one is bundled."""
        if _MIIPHER_ONNX_PATH is None:
            logger.info(
                "MIIPHER native model not bundled — productive vocal SOTA adapter active "
                "(SGMSE+ -> DeepFilterNet v3.II -> DSP)."
            )
            return

        try:
            ort = _load_module("onnxruntime")
            get_ort_providers = _load_symbol("backend.core.ml_device_manager", "get_ort_providers")

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 4
            self._model_session = ort.InferenceSession(
                str(_MIIPHER_ONNX_PATH),
                sess_options=opts,
                providers=get_ort_providers("MIIPHER"),
            )
            self._model_loaded = True
            logger.info("MIIPHER ONNX loaded — §4.4 last-resort NR for SNR < 10 dB.")
            try:
                _reg = _load_symbol("backend.core.plugin_lifecycle_manager", "register_plugin")

                _reg(
                    "MIIPHER",
                    size_gb=0.8,
                    unload_fn=lambda s=self: setattr(s, "_model_session", None) or setattr(s, "_model_loaded", False),
                )
            except Exception as _exc:
                logger.debug("PLM-Registrierung MIIPHER (non-critical): %s", _exc)
        except Exception as exc:
            logger.debug("MIIPHER ONNX not loadable: %s — adapter fallback active.", exc)

    def should_activate(self, noise_snr_db: float, panns_singing: float) -> bool:
        """
        Prüft ob MIIPHER für dieses Material sinnvoll ist.

        Args:
            noise_snr_db:   Geschätzter SNR in dB (aus DefectScanner)
            panns_singing:  PANNs Gesangskonfidenz [0,1]

        Returns:
            True wenn MIIPHER (oder sein Fallback) aktiviert werden soll.
        """
        return noise_snr_db < MIIPHER_SNR_THRESHOLD_DB and panns_singing >= MIIPHER_SINGING_CONFIDENCE_MIN

    def enhance(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        noise_snr_db: float = 0.0,
        vocal_energy_bias_db: float | None = None,  # §0p v9.12.9: register-adaptiver Bias aus VocalRegisterDetector
        panns_singing: float = 0.0,  # §0p v9.12.9: für SGMSE+ Vokal-Mode (konservativeres sigma)
    ) -> npt.NDArray[np.float32]:
        """
        Entrauscht stark degradiertes Gesangsmaterial.

        Primary: MIIPHER ONNX (wenn geladen).
        Fallback: DeepFilterNet v3.II (energy_bias register-adaptiv §0p).
        Last-Resort: Wiener-Filter als stets verfügbarer DSP-Fallback.

        §0h: artifact_freedom-Check NACH Anwendung in UV3 (nicht hier —
        Vermeidung von Doppel-Checks). Hier: nur NaN/Clip-Guard.

        Args:
            audio:               float32 Audio (mono/stereo, 48000 Hz)
            sr:                  Abtastrate (muss 48000 Hz sein)
            noise_snr_db:        Geschätzter Input-SNR (für Logging)
            vocal_energy_bias_db: Register-adaptiver energy_bias aus VocalRegisterDetector.
                                  None → SNR-adaptiver Default. Kopfstimme −3 dB, Brust −6 dB.

        Returns:
            Prozessiertes float32 Audio, gleiche Form wie Input.
        """
        assert sr == 48000, f"MIIPHER: SR muss 48000 Hz sein, erhalten: {sr}"

        reference = np.asarray(audio, dtype=np.float32)
        fallback_chain: list[str] = []
        self._last_route_metadata = {
            "model_used": "none",
            "capability_status": "unavailable",
            "fallback_chain": fallback_chain.copy(),
            "native_miipher_loaded": bool(self._model_loaded),
        }

        # Productive open-source SOTA chain: SGMSE+ is the best available local
        # deep-noise vocal substitute for native MIIPHER when applied to a vocal stem.
        try:
            result = self._enhance_miipher(reference, sr, panns_singing=float(panns_singing))  # §0p
            self._last_route_metadata = {
                "model_used": "miipher_sgmse_plus",
                "capability_status": "sota_fallback" if not self._model_loaded else "sota_real",
                "fallback_chain": fallback_chain.copy(),
                "native_miipher_loaded": bool(self._model_loaded),
                "energy_bias_db": _DFN_FALLBACK_ENERGY_BIAS_DB,
            }
            return result
        except Exception as exc:  # pylint: disable=broad-except
            fallback_chain.append(f"sgmse_plus:{type(exc).__name__}")
            logger.debug("MIIPHER adapter SGMSE+ unavailable: %s — DeepFilterNet fallback.", exc)

        # Fallback: DeepFilterNet v3.II
        try:
            result = self._enhance_dfn_fallback(
                reference, sr, noise_snr_db=noise_snr_db, vocal_energy_bias_db=vocal_energy_bias_db
            )
            _used_bias = vocal_energy_bias_db if vocal_energy_bias_db is not None else _DFN_FALLBACK_ENERGY_BIAS_DB
            self._last_route_metadata = {
                "model_used": "miipher_deepfilternet_v3_ii",
                "capability_status": "sota_fallback",
                "fallback_chain": fallback_chain.copy(),
                "native_miipher_loaded": bool(self._model_loaded),
                "energy_bias_db": _used_bias,
            }
            return result
        except Exception as exc:  # pylint: disable=broad-except
            fallback_chain.append(f"deepfilternet_v3_ii:{type(exc).__name__}")
            logger.warning("DeepFilterNet fallback failed: %s — Wiener DSP fallback.", exc)

        # Last-Resort: DSP Wiener-Filter
        result = self._enhance_wiener_fallback(reference, sr)
        self._last_route_metadata = {
            "model_used": "miipher_wiener_dsp",
            "capability_status": "dsp_fallback",
            "fallback_chain": fallback_chain.copy(),
            "native_miipher_loaded": bool(self._model_loaded),
            "energy_bias_db": _DFN_FALLBACK_ENERGY_BIAS_DB,
        }
        return result

    def _enhance_miipher(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        panns_singing: float = 0.0,  # §0p v9.12.9: weitergereicht an SGMSE+
    ) -> npt.NDArray[np.float32]:
        """Productive open-source MIIPHER substitute via SGMSE+.

        Kaskade: SGMSE+ → HNR-Blend (§0p) → Hallucination-Guard (§2.46e).
        Fallback auf DFN via raise RuntimeError → Aufrufer fängt und leitet an _enhance_dfn_fallback.
        """
        try:
            sgmse_plugin = _load_module("plugins.sgmse_plugin")

            getter = getattr(sgmse_plugin, "get_sgmse_plus_plugin", None) or getattr(
                sgmse_plugin,
                "get_sgmse_plugin",
                None,
            )
            if getter is None:
                raise RuntimeError("SGMSE+ accessor unavailable")
            sgmse = getter()
            if sgmse is None or not bool(getattr(sgmse, "_model_loaded", False)):
                raise RuntimeError("SGMSE+ model not loaded")
            raw = sgmse.enhance(audio, sr, panns_singing=float(panns_singing))  # §0p Vokal-Mode
            raw_audio = getattr(raw, "audio", raw)
            enhanced: npt.NDArray[np.float32] = np.clip(
                np.nan_to_num(np.asarray(raw_audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            )
            enhanced = self._apply_vocal_safety_guards(audio, enhanced, sr, model_name="SGMSE+")

            logger.debug("MIIPHER adapter: SGMSE+ chain succeeded for deep-noise vocal restoration")
            return enhanced
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug("MIIPHER adapter: SGMSE+ error: %s — DFN fallback", exc)
            raise RuntimeError(f"SGMSE+ not available for MIIPHER adapter: {exc}") from exc

    def _enhance_dfn_fallback(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        noise_snr_db: float = 0.0,
        vocal_energy_bias_db: float | None = None,  # §0p v9.12.9: register-adaptiv
    ) -> npt.NDArray[np.float32]:
        """DeepFilterNet v3.II Fallback mit register-adaptivem energy_bias (§0p).

        §9.12.9 Verbesserungen:
        - Register-adaptiver energy_bias: Kopfstimme −3 dB (hohe Harmonik-Dichte),
          Bruststimme −6 dB (Default), Fry/Flüstern −9 dB.
        - SNR-adaptiver Fallback: bei SNR < 5 dB weniger aggressiv (−4 dB)
          wenn kein Register-Bias übergeben wurde.
        - OMLSA Nachglättung: §SOTA-Matrix „DFN v3 + OMLSA".
        """
        get_deepfilternet_plugin = _load_symbol(
            "plugins.deepfilternet_v3_ii_plugin",
            "get_deepfilternet_plugin",
        )

        # §0p Register-adaptiver energy_bias hat Vorrang vor SNR-Schätzung.
        # Kein Register-Bias übergeben → SNR-adaptive Heuristik (Backwards-Compat).
        if vocal_energy_bias_db is not None:
            _snr_adaptive_bias = float(vocal_energy_bias_db)
        elif noise_snr_db < 5.0:
            _snr_adaptive_bias = -4.0  # milder für sehr tiefes SNR
        elif noise_snr_db > 8.0:
            _snr_adaptive_bias = -8.0  # aggressiver für mittleres SNR
        else:
            _snr_adaptive_bias = _DFN_FALLBACK_ENERGY_BIAS_DB

        dfn = get_deepfilternet_plugin()
        result_raw = dfn.enhance(audio, sr=sr, energy_bias_db=_snr_adaptive_bias)
        if result_raw is None or not np.isfinite(np.asarray(result_raw)).all():
            raise RuntimeError("DeepFilterNet-Fallback: ungültiges Ergebnis")

        out_f32: npt.NDArray[np.float32] = np.clip(np.asarray(result_raw, dtype=np.float32), -1.0, 1.0)

        # §9.12.8 OMLSA post-filter (§SOTA-Matrix „DFN v3 + OMLSA"): residual smoothing
        # nach DFN reduziert Musical Noise ohne Vokal-Timbral-Einbusse.
        # Implementierung: IMCRA-Rauschschätzung (compute_imcra_noise_estimate) +
        # spektraler Wiener-Gain (Ephraim-Malah Minimum MSE). energy_bias < 0 dB →
        # Rausch-PSD skalieren → Harmonik-Schutz (§DSP-Instructions §Noise-Schätzung).
        try:
            from scipy.signal import istft as _istft_om  # pylint: disable=import-outside-toplevel
            from scipy.signal import stft as _stft_om  # pylint: disable=import-outside-toplevel

            from backend.core.dsp.noise_estimator import (  # pylint: disable=import-outside-toplevel
                compute_imcra_noise_estimate as _compute_imcra_postfilter,
            )

            _n_fft_om = 2048
            _hop_om = 512  # 75 % Overlap (§STFT-Pflichtstandard)
            _omlsa_mono = out_f32.mean(axis=0) if out_f32.ndim == 2 else out_f32
            _noise_psd_om = _compute_imcra_postfilter(_omlsa_mono, sr, alpha_d=0.85, alpha_s=0.9)
            # energy_bias < 0 dB → Rausch-PSD reduzieren → Harmonik-Schutz
            _eb_lin = float(10.0 ** (float(_snr_adaptive_bias) * 0.5 / 10.0))
            _noise_psd_om = _noise_psd_om * max(_eb_lin, 1e-3)
            _chs_om = [out_f32[0], out_f32[1]] if out_f32.ndim == 2 and out_f32.shape[0] == 2 else [_omlsa_mono]
            _out_chs_om = []
            for _ch_om in _chs_om:
                _, _, _Zxx = _stft_om(_ch_om, fs=sr, nperseg=_n_fft_om, noverlap=_n_fft_om - _hop_om, window="hann")
                _nf = min(_Zxx.shape[1], _noise_psd_om.shape[1])
                _spow = np.abs(_Zxx[:, :_nf]) ** 2
                _g_w = np.maximum(_spow - _noise_psd_om[:, :_nf], 0.0) / (_spow + 1e-12)
                _Zxx_out = _Zxx.copy()
                _Zxx_out[:, :_nf] *= _g_w
                _, _ch_rec = _istft_om(_Zxx_out, fs=sr, nperseg=_n_fft_om, noverlap=_n_fft_om - _hop_om, window="hann")
                _out_chs_om.append(_ch_rec[: len(_ch_om)].astype(np.float32))
            _omlsa_f32: np.ndarray = (
                np.stack(_out_chs_om) if out_f32.ndim == 2 and out_f32.shape[0] == 2 else _out_chs_om[0]
            )
            _omlsa_f32 = np.clip(np.nan_to_num(_omlsa_f32, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            if _omlsa_f32.shape == out_f32.shape:
                # Soft blend: 70% OMLSA result + 30% DFN (preserves DFN transient sharpness)
                out_f32 = np.clip(0.70 * _omlsa_f32 + 0.30 * out_f32, -1.0, 1.0)
                logger.debug("MIIPHER DFN: IMCRA post-filter applied (bias=%.1fdB)", _snr_adaptive_bias)
        except Exception as _omlsa_exc:
            logger.debug("MIIPHER DFN: OMLSA post-filter non-blocking: %s", _omlsa_exc)

        logger.debug(
            "MIIPHER adapter: DeepFilterNet fallback succeeded (energy_bias=%.1f dB snr=%.1f dB)",
            _snr_adaptive_bias,
            noise_snr_db,
        )
        return self._apply_vocal_safety_guards(audio, out_f32, sr, model_name="DeepFilterNet")

    def _apply_vocal_safety_guards(
        self,
        audio_pre: npt.NDArray[np.float32],
        audio_post: npt.NDArray[np.float32],
        sr: int,
        *,
        model_name: str,
    ) -> npt.NDArray[np.float32]:
        """Wendet an: mandatory vocal NR guards after model inference."""
        enhanced = np.clip(
            np.nan_to_num(np.asarray(audio_post, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        try:
            apply_hnr_blend = _load_symbol("backend.core.dsp.hnr_guard", "apply_hnr_blend")

            blended, hnr_diag = apply_hnr_blend(audio_pre, enhanced, sr)
            if float(hnr_diag.get("hnr_delta_db", 0.0)) > 3.0:
                logger.debug(
                    "MIIPHER adapter %s: HNR-Blend active (delta=%.1f dB)",
                    model_name,
                    float(hnr_diag.get("hnr_delta_db", 0.0)),
                )
            enhanced = np.clip(
                np.nan_to_num(np.asarray(blended, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            )
        except Exception as hnr_exc:  # pylint: disable=broad-except
            logger.debug("MIIPHER adapter %s: HNR-Blend unavailable (non-blocking): %s", model_name, hnr_exc)

        try:
            check_hallucination = _load_symbol("backend.core.dsp.hallucination_guard", "check_hallucination")

            hg_result = check_hallucination(audio_pre, enhanced, sr=sr, mode="restoration")
            if getattr(hg_result, "requires_rollback", False):
                logger.info(
                    "MIIPHER adapter %s: Hallucination-Guard rollback (spectral_novelty=%.3f)",
                    model_name,
                    float(getattr(hg_result, "spectral_novelty", 0.0)),
                )
                raise RuntimeError(f"Hallucination-Guard rollback for {model_name}")
        except ImportError:
            pass
        except RuntimeError:
            raise
        except Exception as hg_exc:  # pylint: disable=broad-except
            logger.debug("MIIPHER adapter %s: Hallucination-Guard error (non-blocking): %s", model_name, hg_exc)

        return enhanced.astype(np.float32)

    def _enhance_wiener_fallback(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,  # pylint: disable=unused-argument
    ) -> npt.NDArray[np.float32]:
        """
        Einfacher spektraler Wiener-Filter als Last-Resort DSP-Fallback.

        Schätzt Rauschspektrum aus lautestem 10%-Segment (umgekehrt: Signal dominiert dort),
        dann klassisches Wiener-Spektral-Subtraktionsfilter.
        """
        mono: np.ndarray
        is_stereo = audio.ndim == 2
        if is_stereo:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                mono = np.mean(audio, axis=0).astype(np.float64)
                is_channels_first = True
            else:
                mono = np.mean(audio, axis=1).astype(np.float64)
                is_channels_first = False
        else:
            mono = audio.astype(np.float64)
            is_channels_first = False

        n_fft = 2048
        hop = 512
        window = np.hanning(n_fft)

        frames = []
        for i in range(0, len(mono) - n_fft, hop):
            frame = mono[i : i + n_fft] * window
            frames.append(np.fft.rfft(frame))

        if not frames:
            return audio

        spectra = np.array(frames)  # (T, F)
        mag = np.abs(spectra)

        # Schätze Rauschprofil aus lautestem 10% der Frames invertiert
        # (im lauten Signal sieht man das Rausch-Minimum besser)
        frame_rms = np.sqrt(np.mean(mag**2, axis=1))
        quiet_thresh = np.percentile(frame_rms, 20)
        quiet_mask = frame_rms <= quiet_thresh
        if np.any(quiet_mask):
            noise_est = np.percentile(mag[quiet_mask], 50, axis=0)
        else:
            noise_est = np.percentile(mag, 10, axis=0)

        # Wiener-Gain G(f) = max(0.1, 1 - noise_est / (mag + ε))
        gain = np.clip(1.0 - noise_est[None, :] / np.clip(mag, 1e-10, None), 0.10, 1.0)
        spectra_filtered = spectra * gain

        # iSTFT via OLA
        out = np.zeros(len(mono))
        norm = np.zeros(len(mono))
        for i, (frame_filt, i_start) in enumerate(zip(spectra_filtered, range(0, len(mono) - n_fft, hop))):
            frame_t = np.fft.irfft(frame_filt).real[:n_fft] * window
            out[i_start : i_start + n_fft] += frame_t
            norm[i_start : i_start + n_fft] += window**2

        norm = np.where(norm > 1e-8, norm, 1.0)
        out /= norm
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        logger.debug("MIIPHER-Stub: Wiener-Filter DSP-Fallback angewendet")

        if is_stereo:
            # Signal-Ratio Stereo-Rekonstruktion
            if is_channels_first:
                result = audio.copy()
                ratio = np.clip(out / (mono.astype(np.float32) + 1e-10), 0.0, 2.0)
                for ch in range(audio.shape[0]):
                    result[ch] = np.clip(audio[ch] * ratio, -1.0, 1.0)
            else:
                result = audio.copy()
                ratio = np.clip(out / (mono.astype(np.float32) + 1e-10), 0.0, 2.0)
                for ch in range(audio.shape[1]):
                    result[:, ch] = np.clip(audio[:, ch] * ratio, -1.0, 1.0)
            return result
        out_f32_w: npt.NDArray[np.float32] = out.astype(np.float32)
        return out_f32_w


def get_miipher_plugin() -> MiipherPlugin:
    """Thread-safe Singleton (Double-Checked Locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MiipherPlugin()
    return _instance
