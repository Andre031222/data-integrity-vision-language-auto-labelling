import logging
import os

from flask import Flask

from src.webapp.config import config_map
from src.webapp.extensions import db, login_manager, bcrypt, migrate


def create_app(env: str = "development") -> Flask:
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static",
                static_url_path="/static")

    app.config.from_object(config_map.get(env, config_map["default"]))

    # Basic logging so errors show in console even in debug mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Extensions
    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)

    # Import models so Flask-Migrate sees them AND user_loader is registered
    with app.app_context():
        from src.webapp.models import User, Animal, Analysis, SystemConfig  # noqa

    # Blueprints
    from src.webapp.blueprints.public    import bp as public_bp
    from src.webapp.blueprints.auth      import bp as auth_bp
    from src.webapp.blueprints.dashboard import bp as dashboard_bp
    from src.webapp.blueprints.api       import bp as api_bp
    from src.webapp.blueprints.admin     import bp as admin_bp

    app.register_blueprint(public_bp)
    app.register_blueprint(auth_bp,      url_prefix="/auth")
    app.register_blueprint(dashboard_bp, url_prefix="/dashboard")
    app.register_blueprint(api_bp,       url_prefix="/api/v1")
    app.register_blueprint(admin_bp,     url_prefix="/admin")

    # Context processor: inject site_config into every template
    @app.context_processor
    def inject_site_config():
        try:
            from src.webapp.models.system_config import get_site_config
            return {"site_config": get_site_config()}
        except Exception:
            return {"site_config": {}}

    return app
