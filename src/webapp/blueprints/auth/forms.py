from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from src.webapp.models.user import User


class LoginForm(FlaskForm):
    email    = StringField("Email",      validators=[DataRequired(), Email()])
    password = PasswordField("Contraseña", validators=[DataRequired()])
    remember = BooleanField("Mantener sesión")
    submit   = SubmitField("Ingresar")


class RegisterForm(FlaskForm):
    username = StringField("Usuario",
                           validators=[DataRequired(), Length(3, 64)])
    email    = StringField("Email",
                           validators=[DataRequired(), Email()])
    password = PasswordField("Contraseña",
                             validators=[DataRequired(), Length(8, 128)])
    confirm  = PasswordField("Confirmar contraseña",
                             validators=[DataRequired(), EqualTo("password")])
    submit   = SubmitField("Crear cuenta")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError("Email ya registrado.")

    def validate_username(self, field):
        if User.query.filter_by(username=field.data).first():
            raise ValidationError("Nombre de usuario ya en uso.")
