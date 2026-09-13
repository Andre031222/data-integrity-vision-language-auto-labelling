#!/usr/bin/env python
"""Create or update the AlpacaVision super-admin user."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.webapp import create_app
from src.webapp.extensions import db, bcrypt
from src.webapp.models.user import User


def main() -> None:
    email    = "andrevilcasolorzano@gmail.com"
    username = "andrevilca"
    password = "75521963"
    role     = "super_admin"

    app = create_app("development")
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if user:
            user.password  = bcrypt.generate_password_hash(password).decode("utf-8")
            user.role      = role
            user.is_active = True
            db.session.commit()
            print(f"[OK] Usuario actualizado -> {email}  role={role}")
        else:
            user = User(
                username  = username,
                email     = email,
                password  = bcrypt.generate_password_hash(password).decode("utf-8"),
                role      = role,
                is_active = True,
            )
            db.session.add(user)
            db.session.commit()
            print(f"[OK] Usuario creado -> {email}  role={role}")


if __name__ == "__main__":
    main()
