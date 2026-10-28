# 🚀 Deploying SafeDrive AI to Vercel

This application is **100% client-side enabled with WebAssembly AI (MediaPipe Face Mesh)**, meaning it requires **zero Python backend servers** in production and runs entirely in the user's browser with real-time hardware acceleration.

---

## 🌟 Quick Options to Deploy to Vercel

### Option 1: Deploy via GitHub (Recommended & Free)

1. **Initialize Git & Push to GitHub**:
   If your project isn't on GitHub yet, run in PowerShell inside `E:\sleep_detector`:
   ```powershell
   git init
   git add .
   git commit -m "feat: complete website for vercel deployment"
   ```
   Create a new repository on [GitHub](https://github.com/new) and push your code:
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

2. **Import to Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in (with GitHub).
   - Click **"Add New..."** $\to$ **"Project"**.
   - Select your GitHub repository and click **"Import"**.
   - In **Project Settings**:
     - **Framework Preset**: *Other*
     - **Root Directory**: `./` (leave default)
   - Click **"Deploy"**.

3. **Done!**
   In about 10–20 seconds, your site will be live at `https://your-project-name.vercel.app`.

---

### Option 2: Deploy in 10 Seconds using Vercel CLI

If you have Node.js installed on your computer:

1. Open PowerShell in `E:\sleep_detector`:
   ```powershell
   npx vercel
   ```
2. Follow the quick prompts in your terminal:
   - *Set up and deploy?* Press <kbd>Y</kbd>
   - *Which scope?* Press <kbd>Enter</kbd> (your account)
   - *Link to existing project?* Press <kbd>N</kbd>
   - *Project name?* Press <kbd>Enter</kbd> (or type `safedrive-ai`)
   - *Directory located?* Press <kbd>Enter</kbd> (`./`)
3. When prompted to deploy to production:
   ```powershell
   npx vercel --prod
   ```
4. Vercel will output your instant live URL!

---

## 🧪 Testing Locally in Your Browser

You can preview the exact website locally right now without Vercel:

1. Open PowerShell and run Python's built-in static server:
   ```powershell
   python -m http.server 8000
   ```
2. Open your browser to:
   **http://localhost:8000**

You will see the complete dashboard with webcam AI, 2D highway simulation, audio buzzer, and all controls working immediately.

---

## ⚙️ How It Works on Vercel

| Feature | How it operates on Vercel |
|---|---|
| **Driver AI Face Tracking** | Runs in browser WebAssembly via Google MediaPipe CDN |
| **Privacy** | 100% private. Video never leaves the user's browser or device |
| **Road & Vehicle Simulation** | Runs at 60 FPS on HTML5 Canvas |
| **Audio Buzzer** | Synthesized dynamically using the Web Audio API |
| **Braking Controllers** | Abrupt, Gradual, and Adaptive algorithms executed in JS |
| **Zero Hosting Cost** | Fully compatible with Vercel's free Hobby tier |
