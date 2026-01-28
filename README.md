# AETHER-GEN - Generative Engineering Design Platform

<div align="center">

![Version](https://img.shields.io/badge/version-1.5.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)
![React](https://img.shields.io/badge/react-18+-61DAFB.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA%2012.1-76B900.svg)

**AI-Powered 3D Point Cloud Generation & Engineering Design System**

</div>

---

## 🌟 Overview

AETHER-GEN is a cutting-edge generative engineering platform that combines:
- **Point-E Integration**: OpenAI's text-to-3D point cloud generation with GPU acceleration
- **Real-time 3D Visualization**: Interactive WebGL-based design viewer
- **Engineering Intelligence**: Automated compliance checks, cost estimation, and design auditing
- **CAD Export**: Industry-standard DXF file generation

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **3D Point Cloud Generation** | Generate 3D models from text prompts using Point-E on GPU |
| 🏗️ **Engineering Design** | Create pumps, tanks, pipes, valves, and filters |
| 📊 **Real-time Metrics** | Cost estimation, power calculations, compliance checks |
| 💬 **AI Design Auditor** | Chat with AI about your design for engineering insights |
| 📁 **CAD Export** | Download designs as DXF files for CAD software |
| 🌐 **BIM Support** | IFC file upload and parsing (optional) |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA 12.1 (recommended for Point-E)

### 1. Clone & Setup

```bash
git clone <repository-url>
cd Generative-Design
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend/requirements.txt

# Install PyTorch with CUDA (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the Application

**Terminal 1 - Backend:**
```powershell
cd src/server
$env:PORT=8001; $env:POINT_E_DEVICE="cuda"; $env:USE_POINT_E="true"; python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 5. Open in Browser

Navigate to `http://localhost:5173`

## 🏛️ Project Structure

```
Generative-Design/
├── src/
│   └── server/              # Backend Python source
│       ├── main.py          # FastAPI application
│       ├── engine.py        # Engineering intelligence
│       ├── point_e_service.py # Point-E GPU integration
│       ├── exporter.py      # DXF/PLY/OBJ export
│       ├── bim_handler.py   # IFC file handling
│       └── local_engine.py  # Heuristic design engine
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main application
│   │   └── components/
│   │       ├── DesignViewer.tsx    # 3D scene viewer
│   │       └── PointCloudViewer.tsx # Point cloud renderer
│   └── package.json
├── backend/                 # Legacy backend location
├── .venv/                   # Python virtual environment
└── README.md
```

## 🎮 Usage Guide

### Generate Engineering Designs
1. Enter a design prompt: `"water treatment plant with 3 pumps and 2 storage tanks"`
2. Click **"Generate Design"** - Creates schematic 3D layout
3. Click **"3D Point Cloud"** - Generates AI point cloud (GPU)

### Example Prompts
- `"industrial cooling system with heat exchanger"`
- `"agricultural irrigation system with filters"`
- `"water treatment plant with spherical tanks"`
- `"a red chair"` (for Point-E 3D generation)

### Quality Settings
| Quality | Points | Speed | Use Case |
|---------|--------|-------|----------|
| Fast | 1,024 | ~10s | Quick preview |
| Normal | 4,096 | ~25s | Standard quality |
| High | 4,096 | ~30s | Best quality |

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Backend server port |
| `POINT_E_DEVICE` | auto | `cuda` or `cpu` |
| `USE_POINT_E` | false | Enable Point-E generation |
| `VITE_API_URL` | http://localhost:8000 | Frontend API endpoint |

### Backend `.env`
```env
POINT_E_DEVICE=cuda
USE_POINT_E=true
PORT=8001
```

### Frontend `.env`
```env
VITE_API_URL=http://localhost:8001
```

## 🖥️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/generate` | Generate engineering design |
| POST | `/generate-points` | Generate 3D point cloud |
| POST | `/consult` | AI design consultation |
| POST | `/export-point-cloud` | Export to PLY/OBJ/JSON |
| POST | `/upload-ifc` | Upload BIM/IFC file |

## 🎯 Technology Stack

### Backend
- **FastAPI** - High-performance async API
- **Point-E** - OpenAI's 3D generation model
- **PyTorch + CUDA** - GPU acceleration
- **ezdxf** - CAD file generation

### Frontend
- **React 18** - UI framework
- **Three.js / React Three Fiber** - 3D graphics
- **Framer Motion** - Animations
- **TailwindCSS** - Styling

## 📦 Dependencies

### Backend (requirements.txt)
```
fastapi>=0.100.0
uvicorn>=0.22.0
python-dotenv>=1.0.0
ezdxf>=1.1.0
numpy>=1.24.0
requests>=2.31.0
python-multipart>=0.0.6
point-e
torch
torchvision
torchaudio
```

### Frontend (package.json)
```json
{
  "@react-three/fiber": "^8.x",
  "@react-three/drei": "^9.x",
  "three": "^0.160.x",
  "axios": "^1.x",
  "framer-motion": "^10.x",
  "lucide-react": "^0.x"
}
```

## 🐛 Troubleshooting

### CUDA Not Available
```powershell
# Verify CUDA installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### typing-extensions Error
```bash
pip install --upgrade typing-extensions
```

### Point-E Model Download
First run downloads ~500MB of models. Allow 2-5 minutes.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ using AI-powered engineering**

</div>