import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # project root


class Config:
    SECRET_KEY                    = os.environ.get("SECRET_KEY", "av-dev-secret-change-in-prod")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH             = 16 * 1024 * 1024  # 16 MB
    UPLOAD_FOLDER                  = str(ROOT / "outputs" / "uploads")
    SQLALCHEMY_DATABASE_URI        = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:221203@localhost:5432/alpacavision",
    )


class DevelopmentConfig(Config):
    DEBUG            = True
    SQLALCHEMY_ECHO  = False


class ProductionConfig(Config):
    DEBUG            = False
    SQLALCHEMY_ECHO  = False


config_map = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "default":     DevelopmentConfig,
}
