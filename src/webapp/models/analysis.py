from datetime import datetime, timezone
from src.webapp.extensions import db


class Analysis(db.Model):
    __tablename__ = "analyses"

    id             = db.Column(db.Integer, primary_key=True)
    timestamp      = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    image_path     = db.Column(db.String(256))
    annotated_path = db.Column(db.String(256))

    n_detections    = db.Column(db.Integer, default=0)
    detections_json = db.Column(db.JSON)
    eye_results     = db.Column(db.JSON)
    leg_results     = db.Column(db.JSON)

    has_anomaly    = db.Column(db.Boolean, default=False, index=True)
    severity       = db.Column(db.String(16), default="none")
    findings_json  = db.Column(db.JSON)
    summary        = db.Column(db.Text)
    vet_report     = db.Column(db.Text)
    conf_threshold = db.Column(db.Float, default=0.4)

    user_id   = db.Column(db.Integer, db.ForeignKey("users.id"),   nullable=False)
    animal_id = db.Column(db.Integer, db.ForeignKey("animals.id"), nullable=True)

    user   = db.relationship("User",   back_populates="analyses", foreign_keys=[user_id])
    animal = db.relationship("Animal", back_populates="analyses", foreign_keys=[animal_id])

    def __repr__(self):
        return f"<Analysis {self.id} {'ANOMALY' if self.has_anomaly else 'normal'}>"
