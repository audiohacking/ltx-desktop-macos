# Plan: ltx-mlx — Port MLX standalone de ltx-core + intégration dans LTX Desktop

## Contexte

L'implémentation actuelle du pipeline vidéo/audio LTX-2.3 dans LTX Desktop s'appuie sur ~14 imports de `mlx_video` (package externe) et ~4500 lignes de code vendoré dans `backend/engine/ltx23_model/`. Ce package peut diverger de la référence ltx-core officielle et manque de fonctionnalités (BWE, audio encoder, keyframe conditioning). L'audio est limité à 16kHz avec une qualité bruitée — possiblement dû à des ratios d'upsampling vocoder incorrects.

**Objectif** : Créer un package Python autonome `ltx-mlx` qui porte l'intégralité de ltx-core en MLX pur pour Apple Silicon, puis l'utiliser comme dépendance dans LTX Desktop à la place de mlx_video et du code vendoré.

---

## Architecture

### Nouveau repo : `ltx-mlx`

```
ltx-mlx/
├── pyproject.toml                    # pip install ltx-mlx
├── README.md
├── src/ltx_mlx/
│   ├── __init__.py                   # Version, API publique
│   │
│   ├── model/                        # Transformer DiT
│   │   ├── transformer.py            # BasicAVTransformerBlock
│   │   ├── attention.py              # Multi-head attention + RoPE
│   │   ├── feed_forward.py           # MLP blocks
│   │   ├── model.py                  # LTXModel, X0Model, Modality
│   │   ├── rope.py                   # Rotary position embeddings
│   │   └── timestep_embedding.py     # AdaLayerNorm
│   │
│   ├── vae/                          # Video VAE
│   │   ├── decoder.py                # VideoDecoder (streaming to ffmpeg)
│   │   ├── encoder.py                # VideoEncoder (for I2V, retake, V2V)
│   │   └── patchifier.py             # VideoLatentPatchifier, shapes
│   │
│   ├── audio/                        # Audio complet
│   │   ├── decoder.py                # Audio VAE decoder (latent → mel)
│   │   ├── encoder.py                # Audio VAE encoder (mel → latent) [NEW]
│   │   ├── processor.py              # AudioProcessor STFT + mel filterbank [NEW]
│   │   ├── vocoder.py                # BigVGAN v2 (mel → waveform 16kHz)
│   │   └── bwe.py                    # Bandwidth Extension 16→48kHz [NEW]
│   │
│   ├── text_encoder/                 # Text encoding
│   │   ├── language_model.py         # Gemma 3 wrapper via mlx-lm
│   │   ├── connector.py              # Embeddings1DConnector (RoPE refinement)
│   │   └── feature_extractor.py      # GemmaFeaturesExtractorV2
│   │
│   ├── conditioning/                 # Conditioning system
│   │   ├── latent.py                 # LatentState, VideoConditionByLatentIndex
│   │   ├── keyframe.py               # VideoConditionByKeyframeIndex [NEW]
│   │   └── reference.py              # VideoConditionByReferenceLatent [NEW]
│   │
│   ├── pipeline/                     # Pipelines de génération
│   │   ├── denoise.py                # Boucle de diffusion Euler
│   │   ├── generate.py               # API haut niveau generate()
│   │   └── scheduler.py              # Sigma schedules (distilled, full)
│   │
│   ├── upsampler/                    # Latent 2× spatial upscaler
│   │   └── upsampler.py              # LatentUpsampler + upsample_latents()
│   │
│   └── utils/                        # Utilitaires
│       ├── ffmpeg.py                 # find_ffmpeg, find_ffprobe, probe_video_info
│       ├── image.py                  # prepare_image_for_encoding
│       ├── weights.py                # Weight loading, quantization, split safetensors
│       └── memory.py                 # aggressive_cleanup, memory stats
│
├── scripts/
│   └── convert_weights.py            # PyTorch → MLX conversion
│
└── tests/
    ├── test_conditioning.py
    ├── test_pipeline.py
    ├── test_vocoder.py
    ├── test_bwe.py
    └── test_vae.py
```

### Intégration dans LTX Desktop

```python
# Avant (14 imports mlx_video + 4500 lignes vendorées)
from mlx_video.conditioning.latent import LatentState, VideoConditionByLatentIndex, ...
from mlx_video.models.ltx.upsampler import LatentUpsampler
from mlx_video.models.ltx.video_vae.encoder import VideoEncoder
from mlx_video.models.ltx.audio_vae import AudioDecoder
from mlx_video.models.ltx.text_encoder import LanguageModel

# Après (imports ltx_mlx unifiés)
from ltx_mlx.conditioning.latent import LatentState, VideoConditionByLatentIndex, ...
from ltx_mlx.upsampler import LatentUpsampler, upsample_latents
from ltx_mlx.vae.encoder import VideoEncoder, load_vae_encoder
from ltx_mlx.audio.decoder import AudioDecoder, load_audio_decoder
from ltx_mlx.audio.vocoder import load_vocoder, VocoderWithBWE
from ltx_mlx.text_encoder import LanguageModel
from ltx_mlx.pipeline import generate, GenerationConfig, GenerationOutput
from ltx_mlx.utils.memory import aggressive_cleanup
```

L'app desktop supprime `backend/engine/ltx23_model/` (18 fichiers) et le remplace par `ltx-mlx` comme dépendance dans `pyproject.toml`. Le code d'orchestration (`generate_v23.py`, `mlx_runner.py`, `pipelines/`) reste dans l'app — seul le moteur d'inférence MLX migre.

---

## Phases d'implémentation

### Sprint 1 — Bootstrap ltx-mlx + Conditioning + Pipeline (3-4j)

Créer le repo, la structure de package, et porter les composants sans poids réseau.

**1a. Bootstrap repo**
- `pyproject.toml` avec deps : `mlx>=0.31.0`, `mlx-lm>=0.31.0`, `numpy`, `safetensors`, `huggingface-hub`
- Structure de répertoires, `__init__.py` avec version
- CI basique (lint ruff, type check)

**1b. Conditioning system** (port de `mlx_video.conditioning.latent`)
- `src/ltx_mlx/conditioning/latent.py`
  - `LatentState` dataclass
  - `VideoConditionByLatentIndex`
  - `create_initial_state()`, `apply_conditioning()`, `apply_denoise_mask()`, `add_noise_with_state()`
- Opérations tenseur pures MLX, aucun poids

**1c. Pipeline / denoise loop** (port de `ltx23_model/pipeline.py`)
- `src/ltx_mlx/pipeline/denoise.py` — boucle Euler distillée
- `src/ltx_mlx/pipeline/scheduler.py` — `DISTILLED_SIGMAS`, `STAGE_2_SIGMAS`
- Import conditioning depuis le module local

**1d. Utils**
- `src/ltx_mlx/utils/ffmpeg.py` — port de `engine/ffmpeg_utils.py`
- `src/ltx_mlx/utils/memory.py` — `aggressive_cleanup()`, `get_memory_stats()`
- `src/ltx_mlx/utils/image.py` — `prepare_image_for_encoding()`

**Test** : Tests unitaires conditioning (create state, apply mask, verify shapes)

### Sprint 2 — Transformer + Patchifier (3-4j)

Port de l'architecture du modèle (le plus gros en lignes de code).

- `src/ltx_mlx/model/` — port des 6 fichiers de `ltx23_model/` :
  - `model.py` (LTXModel, X0Model, Modality) — 758 lignes
  - `transformer.py` (BasicAVTransformerBlock) — 250 lignes
  - `attention.py` (multi-head + RoPE) — 118 lignes
  - `feed_forward.py` — 60 lignes
  - `rope.py` — 210 lignes
  - `timestep_embedding.py` — 100 lignes
- `src/ltx_mlx/vae/patchifier.py` — VideoLatentPatchifier, AudioPatchifier, shapes
- `src/ltx_mlx/utils/weights.py` — weight loading, quantization config, split safetensors

**Test** : Charger le modèle quantizé, forward pass sur dummy input, vérifier shapes de sortie

### Sprint 3 — VAE (decoder + encoder) + Text Encoder (3-4j)

**3a. Video VAE**
- `src/ltx_mlx/vae/decoder.py` — port de `vae_decoder.py` (streaming decode to ffmpeg)
- `src/ltx_mlx/vae/encoder.py` — port de `vae_encoder.py` (image/video encoding)

**3b. Audio VAE decoder**
- `src/ltx_mlx/audio/decoder.py` — port de `audio_decoder.py`

**3c. Text encoder**
- `src/ltx_mlx/text_encoder/language_model.py` — wrapper Gemma 3 via mlx-lm (remplace `mlx_video.models.ltx.text_encoder.LanguageModel`)
- `src/ltx_mlx/text_encoder/connector.py` — port de `connector.py` (Embeddings1DConnector)
- `src/ltx_mlx/text_encoder/feature_extractor.py` — port de `text_encoder.py` (GemmaFeaturesExtractorV2)

**Test** : Encoder texte → embeddings, encoder image → latent, decoder latent → pixels. Comparer numériquement avec l'implémentation actuelle.

### Sprint 4 — Vocoder + BWE (4-5j) ⭐ Impact audio majeur

**4a. Vérifier/corriger vocoder**
- Inspecter shapes poids `vocoder.safetensors` pour déterminer les vrais ratios d'upsampling
- Si discordance avec `[5,2,2,2,2,2]` actuel → corriger
- `src/ltx_mlx/audio/vocoder.py` — port corrigé de BigVGAN v2

**4b. BWE (Bandwidth Extension)** [NOUVEAU]
- `src/ltx_mlx/audio/bwe.py`
  - `MelSpectrogram` — STFT + mel filterbank (signal processing)
  - `UpSample1d` — rééchantillonnage kaiser-sinc 3× (16→48kHz)
  - `VocoderWithBWE` — base vocoder → upsample → mel → BWE generator → residual
- Poids déjà dans `vocoder.safetensors` (préfixes `vocoder.bwe.*`, `vocoder.mel.*`)

**Test** : Comparer qualité audio 16kHz vs 48kHz, mesurer impact mémoire sur 32GB

### Sprint 5 — Upsampler + Audio Encoder (3-4j)

**5a. Latent Upsampler**
- `src/ltx_mlx/upsampler/upsampler.py` — LatentUpsampler (Conv3d ResBlocks + spatial 2×)
- `upsample_latents()` — denormalize → upsample → renormalize
- Déplacer la logique de chargement poids de `generate_v23.py` vers le module

**5b. Audio Encoder** [NOUVEAU]
- `src/ltx_mlx/audio/encoder.py` — miroir du decoder (pour A2V, retake audio)
- `src/ltx_mlx/audio/processor.py` — AudioProcessor STFT (réutilise MelSpectrogram du Sprint 4)
- Pipeline : waveform → mel → latent

**Test** : Round-trip audio encode→decode, two-stage upscale shape check

### Sprint 6 — Intégration LTX Desktop + Cleanup (3-4j)

**6a. Remplacer imports dans l'app desktop**
- `generate_v23.py` — remplacer 9 imports mlx_video + code vendoré par ltx_mlx
- `encode_text_subprocess.py` — remplacer 2 imports
- `pipeline.py` (app) — utiliser `ltx_mlx.pipeline` ou garder la boucle locale avec `ltx_mlx.conditioning`
- `pipelines/retake.py`, `pipelines/extend.py` — adapter si nécessaire

**6b. Supprimer le code vendoré**
- Supprimer `backend/engine/ltx23_model/` (18 fichiers, ~4500 lignes)
- Supprimer `mlx-video-with-audio` de `pyproject.toml`
- Ajouter `ltx-mlx` comme dépendance

**6c. Tests d'intégration**
- T2V, I2V, preview, retake, extend — bout en bout
- Audio avec BWE
- Marathon test (10 gens 97f@768×512)

### Sprint 7 — Keyframe Conditioning + Features avancées (2-3j) — Phase 2 app

- `src/ltx_mlx/conditioning/keyframe.py` — VideoConditionByKeyframeIndex (tokens appendés + attention mask structuré)
- `src/ltx_mlx/conditioning/reference.py` — VideoConditionByReferenceLatent (IC-LoRA)
- TemporalRegionMask amélioré (audio + vidéo séparés)

---

## Graphe de dépendances

```
Sprint 1 (Bootstrap + Conditioning + Pipeline + Utils)
    │
Sprint 2 (Transformer + Patchifier)
    │
Sprint 3 (VAE + Text Encoder) ──── Sprint 5a (Upsampler)
    │
Sprint 4 (Vocoder + BWE) ───────── Sprint 5b (Audio Encoder)
    │
Sprint 6 (Intégration Desktop + Cleanup)
    │
Sprint 7 (Keyframe Conditioning)
```

Sprints 1→2→3 sont séquentiels (chaque sprint dépend du précédent).
Sprint 4 et 5a peuvent commencer dès Sprint 3 terminé.
Sprint 5b bénéficie du STFT de Sprint 4.
Sprint 6 nécessite tous les sprints précédents.

---

## Ce qui migre vs ce qui reste dans l'app

### Migre vers ltx-mlx (moteur d'inférence)
| Composant | Fichier actuel | Module ltx-mlx |
|-----------|----------------|----------------|
| Transformer | `ltx23_model/model.py` | `ltx_mlx.model.model` |
| Attention + RoPE | `ltx23_model/attention.py`, `rope.py` | `ltx_mlx.model.attention`, `.rope` |
| Diffusion loop | `ltx23_model/pipeline.py` | `ltx_mlx.pipeline.denoise` |
| Video VAE | `ltx23_model/vae_decoder.py`, `vae_encoder.py` | `ltx_mlx.vae.*` |
| Audio VAE + Vocoder | `ltx23_model/audio_decoder.py`, `vocoder.py` | `ltx_mlx.audio.*` |
| Text encoder | `ltx23_model/connector.py`, `text_encoder.py` | `ltx_mlx.text_encoder.*` |
| Conditioning | (mlx_video imports) | `ltx_mlx.conditioning.*` |
| Patchifier | `ltx23_model/patchifier.py` | `ltx_mlx.vae.patchifier` |
| Weight loading | `ltx23_model/loader.py` | `ltx_mlx.utils.weights` |

### Reste dans l'app desktop (orchestration)
| Composant | Fichier | Raison |
|-----------|---------|--------|
| Subprocess orchestration | `mlx_runner.py` | Spécifique à l'archi 2-subprocess de l'app |
| Generation subprocess | `generate_v23.py` | Orchestre le pipeline, appelle ltx_mlx |
| Text encoding subprocess | `encode_text_subprocess.py` | Gestion mémoire 32GB spécifique |
| Pipeline wrappers | `pipelines/t2v.py`, `i2v.py`, etc. | Logique métier app (queue, progress, history) |
| Memory manager | `memory_manager.py` | Spécifique au pattern reload/cleanup de l'app |
| Model manager | `model_manager.py` | Download HF, cache, sélection variante |
| ffmpeg mux | Partie de `generate_v23.py` | Spécifique au format de sortie app |

---

## Estimation totale

| Sprint | Composants | Complexité | Durée |
|--------|-----------|------------|-------|
| 1 | Bootstrap + Conditioning + Pipeline + Utils | LOW-MED | 3-4j |
| 2 | Transformer + Patchifier | MED | 3-4j |
| 3 | VAE + Text Encoder | MED | 3-4j |
| 4 | Vocoder + BWE | HIGH | 4-5j |
| 5 | Upsampler + Audio Encoder | MED-HIGH | 3-4j |
| 6 | Intégration Desktop + Cleanup | MED | 3-4j |
| 7 | Keyframe Conditioning | MED | 2-3j |
| **Total** | | | **~22-28j** |

## Vérification

Après chaque sprint :
1. Tests unitaires dans `ltx-mlx/tests/`
2. Lint (`ruff check`) + type check
3. Après Sprint 6 : régression complète dans l'app desktop
   - T2V 97f@768×512
   - I2V avec image de référence
   - Preview rapide 384×256
   - Retake segment
   - Extend forward/backward
   - Audio avec BWE (comparaison 16kHz vs 48kHz)
   - Marathon test (10 générations consécutives, stable timing, pas d'OOM)
