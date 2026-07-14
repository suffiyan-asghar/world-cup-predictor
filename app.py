import gradio as gr
import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load("model.pkl")
elo_df = pd.read_csv("elo_ratings.csv")
form_df = pd.read_csv("team_form.csv")

teams = sorted(elo_df['team'].unique())
features = ['elo_diff', 'neutral_venue', 'home_form_gf', 'home_form_ga', 'away_form_gf', 'away_form_ga']

def get_elo(team):
    row = elo_df[elo_df['team'] == team]
    return row['elo'].values[0] if not row.empty else 1500

def get_form(team):
    row = form_df[form_df['team'] == team]
    if not row.empty:
        return row['form_gf'].values[0], row['form_ga'].values[0]
    return form_df['form_gf'].mean(), form_df['form_ga'].mean()

def predict_match(team_a, team_b, neutral):
    elo_diff = get_elo(team_a) - get_elo(team_b)
    form_gf_a, form_ga_a = get_form(team_a)
    form_gf_b, form_ga_b = get_form(team_b)

    row = pd.DataFrame([{
        'elo_diff': elo_diff,
        'neutral_venue': 1 if neutral else 0,
        'home_form_gf': form_gf_a,
        'home_form_ga': form_ga_a,
        'away_form_gf': form_gf_b,
        'away_form_ga': form_ga_b
    }])

    prob_a = model.predict_proba(row[features])[0, 1]
    return f"{team_a}: {prob_a:.1%}  |  {team_b}: {1 - prob_a:.1%}"

demo = gr.Interface(
    fn=predict_match,
    inputs=[
        gr.Dropdown(teams, label="Team A", value="Argentina"),
        gr.Dropdown(teams, label="Team B", value="France"),
        gr.Checkbox(label="Neutral venue?", value=True)
    ],
    outputs=gr.Textbox(label="Win Probability"),
    title="World Cup 2026 Match Predictor",
    description="Elo + recent form based logistic regression model, trained on 150+ years of international football results."
)

demo.launch()