"""Key-value system_config table for white-label platform settings."""
from datetime import datetime, timezone
from src.webapp.extensions import db

# --- DEFAULT CONFIG ------------------------------------------------------------
DEFAULT_CONFIG: dict[str, str] = {
    # Identity
    "site_name":         "AlpacaVision AI",
    "site_badge":        "AV",                   # max 7 chars, shown in logo box
    "site_tagline":      "Detección inteligente de anomalías en alpacas",
    # Hero
    "hero_title":        "Diagnóstico veterinario\nasistido por IA",
    "hero_description":  "Detecta anomalías morfológicas en alpacas mediante visión computacional. YOLOv11 + EfficientNet-B2 + Groq Vision con reportes en español.",
    "hero_cta_primary":  "Comenzar gratis",
    "hero_cta_secondary":"Ver cómo funciona",
    "hero_badge_text":   "Sistema activo -- UNA Puno * Semillero Hopfield IIICCD",
    # Metrics
    "metric_1_value":    "0.913",
    "metric_1_label":    "mAP50 Detector",
    "metric_2_value":    "0.833",
    "metric_2_label":    "F1 Clasificador",
    "metric_3_value":    "89.9%",
    "metric_3_label":    "Recall anomalías",
    "metric_4_value":    "5K+",
    "metric_4_label":    "Imágenes entrenadas",
    # Logos / images (empty = use default)
    "logo_url":          "",
    "favicon_url":       "",
    "institution_logo_url": "",
    "cover_image_url":   "",
    # Navigation (JSON array)
    "nav_items": '[{"label":"Características","href":"#features"},{"label":"Cómo funciona","href":"#how"},{"label":"Acerca de","href":"#about"}]',
    # Footer
    "footer_description": "Sistema de visión computacional para detección de anomalías en camélidos sudamericanos.",
    "footer_institution":  "UNA Puno -- Semillero John J. Hopfield * IIICCD",
    "footer_contact":      "",
    "footer_copyright":    "2025 AlpacaVision AI",
    "social_twitter":      "",
    "social_github":       "",
    "social_instagram":    "",
    "social_facebook":     "",
    # Announcement banner
    "announcement_enabled": "false",
    "announcement_text":    "",
    "announcement_color":   "brand",   # brand | red | amber | blue
    # Maintenance
    "maintenance_mode":    "false",
    "maintenance_message": "Sistema en mantenimiento. Volvemos pronto.",
    # Analytics
    "ga4_id": "",
}

# Keys exposed to the public (unauthenticated) /api/v1/public/config endpoint
PUBLIC_KEYS = frozenset({
    "site_name", "site_badge", "site_tagline",
    "hero_title", "hero_description", "hero_cta_primary", "hero_cta_secondary", "hero_badge_text",
    "metric_1_value", "metric_1_label", "metric_2_value", "metric_2_label",
    "metric_3_value", "metric_3_label", "metric_4_value", "metric_4_label",
    "logo_url", "favicon_url", "institution_logo_url", "cover_image_url",
    "nav_items",
    "footer_description", "footer_institution", "footer_contact", "footer_copyright",
    "social_twitter", "social_github", "social_instagram", "social_facebook",
    "announcement_enabled", "announcement_text", "announcement_color",
    "maintenance_mode", "maintenance_message",
    "ga4_id",
})


class SystemConfig(db.Model):
    __tablename__ = "system_config"

    id           = db.Column(db.Integer, primary_key=True)
    config_key   = db.Column(db.String(128), unique=True, nullable=False, index=True)
    config_value = db.Column(db.Text, nullable=True)
    config_type  = db.Column(db.String(32), default="text")   # text | bool | json | image
    updated_at   = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                              onupdate=lambda: datetime.now(timezone.utc))

    # -- class helpers -------------------------------------------------------
    @classmethod
    def get(cls, key: str, default: str | None = None) -> str | None:
        row = cls.query.filter_by(config_key=key).first()
        return row.config_value if row else default

    @classmethod
    def set(cls, key: str, value: str, config_type: str = "text") -> None:
        row = cls.query.filter_by(config_key=key).first()
        if row:
            row.config_value = value
            row.updated_at   = datetime.now(timezone.utc)
        else:
            row = cls(config_key=key, config_value=value, config_type=config_type)
            db.session.add(row)
        db.session.commit()

    def __repr__(self) -> str:
        return f"<SystemConfig {self.config_key}>"


def get_site_config() -> dict[str, str]:
    """Return merged config: DEFAULT_CONFIG overridden by DB values."""
    config = dict(DEFAULT_CONFIG)
    try:
        for row in SystemConfig.query.all():
            config[row.config_key] = row.config_value or ""
    except Exception:
        pass
    return config
