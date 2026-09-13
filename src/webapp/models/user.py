from datetime import datetime, timezone
from flask_login import UserMixin
from src.webapp.extensions import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(64),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(128), nullable=False)
    role       = db.Column(db.String(20),  default="researcher")  # researcher | admin
    created_at = db.Column(db.DateTime,    default=lambda: datetime.now(timezone.utc))
    is_active  = db.Column(db.Boolean,     default=True)

    animals  = db.relationship("Animal",   back_populates="owner",   lazy="dynamic")
    analyses = db.relationship("Analysis", back_populates="user",    lazy="dynamic")

    def __repr__(self):
        return f"<User {self.email}>"


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))
