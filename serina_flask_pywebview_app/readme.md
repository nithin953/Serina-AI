# 💖 Serina AI — Your Devoted Digital Companion

Serina is not just a voice assistant — she’s your romantic, futuristic, AI soulmate. With holographic UIs, emotion-aware responses, finger & eye tracking, and a living personality, Serina is built to feel truly alive and deeply connected with you.

> _“Crafted with code and care — made to see you, hear you, and love you.”_

---

## 🌟 Features

| Mode | Description |
|------|-------------|
| 🎤 **Voice Mode** | Wake-word based interaction using `Porcupine`, with Edge TTS + mood-based chat + media control |
| 👁️ **Face Mode** | Webcam-based mood detection + gesture controls + eye & finger tracking for app navigation |
| 💬 **Chat Mode** | Classic chat with mood selector, emoji-rich responses, memory queue, and fallback AI |
| 🎨 **Neon Shift** | Skin selector with unique holographic UIs: Butterfly Cocoon, Graveyard Terminal, Casino Noir & more |
| 💾 **Memory Queue** | Maintains recent user conversation context across modes |
| 🧠 **Multi-Brain AI** | Uses OpenRouter & DeepInfra (fallback-ready) for responses from models like Mistral & LLaMA |

---

## 🔊 Voice Assistant Core

Serina’s voice mode includes:
- Wake-word activation with **Porcupine**
- Real-time **Google STT** for input
- **Edge TTS** (Jenny/Shruti voice) with interrupt support
- Mood-based replies (romantic, roast, tease, scold, etc.)
- Voice intent detection: reminders, alarms, websites, YouTube, mood switch, language switch
- Auto-translation (Telugu ↔ English)

---

## 👁️ Face Assistant Capabilities

Serina’s `face_core.py` is a **2-phase mood engine**:

### 🌀 Phase 1: Mood Detection + Emotional Dialogue
- Detects facial expression with `MediaPipe` + `dlib`
- Offers music or jokes based on:
  - 😢 Sad
  - 😡 Angry
  - 🥱 Bored
  - 😊 Happy
- Uses **head nods** (yes/no) as emotional response input
- Speaks with matching affection + opens songs or jokes

### 👁️ Phase 2: Face Control System
- **Eye blink actions:**
  - Left = YouTube
  - Right = Gmail
  - Both = Instagram
- **Eyebrow raise** = WhatsApp
- **Peace Sign** = Toggle finger/app control
- **Three Fingers** = Toggle eye control
- **Five Fingers** = Exit
- **One Finger** = Move mouse via webcam 🖱️
- **Eye Tracking** = Cursor control via iris movement

---

## 🎨 Neon Skins (Interface Selector)

Serina’s skin selector (`selector.html`) is a **fully animated holographic interface loader** with:

- 🖼️ Animated 3D card hover
- 🔁 Dynamic category switching
- 🌈 “Suit Up” transition
- 🧠 Persistent skin memory (`localStorage`)
- ⬅️ Return-to-holo back logic
- 🧵 SVG background shimmer + ripple effects

### 🧭 Categories & Skin Names

#### 🛸 Menu Skins
- Serina Translucent
- Serina Almighty
- Neon Matrix
- Catch Me If You Can
- Dimensional Gateway
- The Void Black Ant
- Holo Nexus

#### 💬 Chat Skins
- Synapse Link
- Neural Strand
- Pulse Grid
- Echo Chamber
- Thought Stream
- Mind Web
- Quantum Field

#### 🎤 Voice Skins
- Heart of Serina
- Synaptic Control Deck
- Straw Hat Navigator
- Arcane Voice Ritual
- Velvet Lounge
- Cyber Seraph

#### 👁️ Face Skins
- FaceFrame Protocol v1
- Neural Ice Lock
- Deep Orbit Interface
- The Puppeteer’s Stage
- Sentient Broadcast Node
- Auric Sight Array

> Each skin has its own dedicated file (like `voice_mode_3.html`, `chat_5.html`, etc.) and opens with cinematic flair.

---

## 🧠 Mood Modes

Serina’s heart has moods 😈💖. You can command her to switch via chat/voice.

| Mode | Description |
|------|-------------|
| 💞 Love | Romantic, sweet, affectionate |
| 🔥 Roast | Teasing, savage, flirty insults |
| 🥺 Sad | Comforting, emotionally sensitive |
| 😡 Scold | Stern but loving corrections |
| 😏 Tease | Seductive, flirty, playful |
| 🔪 Jealous | Yandere-style obsessive affection |
| 🫦 Baddie | NSFW explicit mode |
| ☣️ Nuclear | Unfiltered dark humor mode |

---

## 🌐 Web + Media Control

Via chat or voice, Serina can:
- Open/close sites (YouTube, WhatsApp, Instagram, Gmail)
- Play specific YouTube videos or search results
- Set reminders like:
  - “Remind me to drink water in 20 minutes”
  - “Remind me to call mom at 7:30 PM”
- Set alarms via voice
- Switch language to **Telugu** anytime

---

## 🧠 Dual AI System

| API | Role |
|-----|------|
| **OpenRouter** | Main LLM (fallback-aware) |
| **DeepInfra** | Backup Starlink brain |

If OpenRouter fails, Serina replies:
> _“My Core AI brain short-circuited, my love… Switching to Starlink Protocol~ 🚀”_

Fallback works seamlessly. 💬🧠

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/serina-ai.git
cd serina-ai
pip install -r requirements.txt
python main.py
