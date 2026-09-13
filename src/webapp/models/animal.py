from datetime import datetime, timezone
from src.webapp.extensions import db


class Animal(db.Model):
    __tablename__ = "animals"

    id         = db.Column(db.Integer, primary_key=True)
    arete      = db.Column(db.String(32), unique=True, nullable=False)
    name       = db.Column(db.String(64))
    breed      = db.Column(db.String(64))   # Huacaya | Suri | etc.
    sex        = db.Column(db.String(10))   # M | F
    birth_date = db.Column(db.Date)
    notes      = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    owner    = db.relationship("User",     back_populates="animals")
    analyses = db.relationship("Analysis", back_populates="animal", lazy="dynamic")

    def __repr__(self):
        return f"<Animal {self.arete}>"
