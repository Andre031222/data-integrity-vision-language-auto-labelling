from flask import render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from src.webapp.extensions import db
from src.webapp.models.analysis import Analysis
from src.webapp.models.animal import Animal
from . import bp


@bp.route("/")
@login_required
def index():
    recent = (Analysis.query
              .filter_by(user_id=current_user.id)
              .order_by(Analysis.timestamp.desc())
              .limit(5).all())
    stats = {
        "total":    Analysis.query.filter_by(user_id=current_user.id).count(),
        "anomaly":  Analysis.query.filter_by(user_id=current_user.id, has_anomaly=True).count(),
        "animals":  Animal.query.filter_by(owner_id=current_user.id).count(),
    }
    return render_template("dashboard/index.html", recent=recent, stats=stats)


@bp.route("/animals")
@login_required
def animals():
    animals_list = (Animal.query
                    .filter_by(owner_id=current_user.id)
                    .order_by(Animal.created_at.desc()).all())
    return render_template("dashboard/animals.html", animals=animals_list)


@bp.route("/animals/new", methods=["GET", "POST"])
@login_required
def new_animal():
    if request.method == "POST":
        arete = request.form.get("arete", "").strip()
        if not arete:
            flash("El arete es obligatorio.", "error")
            return redirect(url_for("dashboard.new_animal"))
        if Animal.query.filter_by(arete=arete).first():
            flash("Ese número de arete ya existe.", "error")
            return redirect(url_for("dashboard.new_animal"))
        animal = Animal(
            arete=arete,
            name=request.form.get("name", "").strip() or None,
            breed=request.form.get("breed", "").strip() or None,
            sex=request.form.get("sex", "").strip() or None,
            notes=request.form.get("notes", "").strip() or None,
            owner_id=current_user.id,
        )
        db.session.add(animal)
        db.session.commit()
        flash(f"Animal {arete} registrado.", "success")
        return redirect(url_for("dashboard.animals"))
    return render_template("dashboard/new_animal.html")


@bp.route("/animals/<int:animal_id>")
@login_required
def animal_detail(animal_id):
    animal = db.get_or_404(Animal, animal_id)
    if animal.owner_id != current_user.id:
        flash("Sin acceso.", "error")
        return redirect(url_for("dashboard.animals"))
    analyses = (animal.analyses
                .order_by(Analysis.timestamp.desc())
                .limit(20).all())
    return render_template("dashboard/animal_detail.html", animal=animal, analyses=analyses)


@bp.route("/history")
@login_required
def history():
    page     = request.args.get("page", 1, type=int)
    analyses = (Analysis.query
                .filter_by(user_id=current_user.id)
                .order_by(Analysis.timestamp.desc())
                .paginate(page=page, per_page=20, error_out=False))
    return render_template("dashboard/history.html", analyses=analyses)


@bp.route("/profile")
@login_required
def profile():
    return render_template("dashboard/profile.html")
