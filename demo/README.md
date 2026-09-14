# paqocha — AlpacaVision public demo

A single-page research demo for the scene-deduplicated **YOLOv11n** alpaca detector. Upload a
photo → get the annotated image with bounding boxes and a detections table. Academic /
institutional UI, bilingual (EN/ES), light + dark themes. No login, no database.

The ocular classifier is intentionally **not** served (it predicts its own auto-labels
above chance, but those labels carry no validated clinical meaning, under the
protocol; see the paper's honest negative result). An **optional, experimental image
description** is produced by a **local** vision model via Ollama — the image never leaves the
server, and it is clearly labelled as *not a veterinary diagnosis*.

Live demo: **https://paqocha.ginit.dev**

Endpoints: `GET /health`, `POST /api/predict` (fast detection), `POST /api/describe`
(optional local description, called separately so detection stays instant).

## Run locally

```bash
# from the repository root
python -m venv .venv-demo && source .venv-demo/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r demo/requirements.txt

python demo/app.py            # http://localhost:5070
```

The detector weights (`models/detector/best_v3_n.pt`) must be present locally
(distributed via GitHub Releases / Zenodo).

## Optional: local image description (Ollama)

The experimental description is produced by a local vision model — **private, no data leaves
the machine**. It degrades gracefully to nothing if Ollama is not running, so the demo works
without it.

```bash
# install Ollama (https://ollama.com), then pull a small vision model
ollama pull moondream          # ~1.7 GB, runs on CPU or GPU
```

The app talks to Ollama at `http://127.0.0.1:11434` by default. Override with env vars:
`OLLAMA_URL`, `OLLAMA_VISION_MODEL` (e.g. `moondream` or `qwen2.5vl:3b`). On CPU the
description takes ~30–60 s, so it is fetched by a **separate** `/api/describe` call and the
detection stays instant. The input image is downscaled to 512 px before inference.

## Deploy to production (paqocha.ginit.dev)

Same pattern as `frost.ginit.dev` — Gunicorn behind Nginx, TLS via Let's Encrypt.

```bash
# 1) On the server, clone into /opt/paqocha and build the env
sudo mkdir -p /opt/paqocha && cd /opt/paqocha
git clone https://github.com/Andre031222/data-integrity-vision-language-auto-labelling.git
cd data-integrity-vision-language-auto-labelling
python -m venv .venv-demo && . .venv-demo/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r demo/requirements.txt
# place best_v3_n.pt in models/detector/ (from Zenodo / Releases)

# 2) systemd service (Gunicorn on 127.0.0.1:9060)
sudo cp demo/deploy/paqocha.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now paqocha

# 3) Nginx reverse proxy + DNS + TLS
sudo cp demo/deploy/nginx-paqocha.conf /etc/nginx/sites-available/paqocha.ginit.dev
sudo ln -s /etc/nginx/sites-available/paqocha.ginit.dev /etc/nginx/sites-enabled/
#   add DNS A record: paqocha.ginit.dev -> server IP
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d paqocha.ginit.dev
```

Check it: `curl -s https://paqocha.ginit.dev/health` → `{"status":"ok","detector":true}`.

## Files

```
demo/
├── app.py                     Flask app: / + /api/predict + /api/describe + /health
├── templates/index.html       single-page UI (self-contained)
├── requirements.txt           minimal CPU inference stack
├── gunicorn_conf.py           Gunicorn settings (127.0.0.1:9060)
└── deploy/
    ├── paqocha.service        systemd unit
    └── nginx-paqocha.conf     Nginx reverse-proxy site
```
