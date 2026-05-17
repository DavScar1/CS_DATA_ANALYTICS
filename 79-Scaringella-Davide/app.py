from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import io
import base64

app = Flask(__name__)

# =============================================
# Load and clean the dataset once at startup
# =============================================
df = pd.read_csv('Mental_Health_Lifestyle_Dataset.csv')
df = df.drop_duplicates()
df = df.dropna(subset=['Sleep Hours', 'Stress Level', 'Mental Health Condition', 'Happiness Score'])
df['Gender'] = df['Gender'].str.lower().str.strip()
df['Mental Health Condition'] = df['Mental Health Condition'].str.lower().str.strip()
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df = df[(df['sleep_hours'] >= 0) & (df['sleep_hours'] <= 24)]
df = df[(df['happiness_score'] >= 0) & (df['happiness_score'] <= 10)]

# Colour palette for mental health conditions
palette = {
    'none':       '#2ecc71',
    'anxiety':    '#f39c12',
    'depression': '#e74c3c',
    'bipolar':    '#9b59b6',
    'ptsd':       '#e67e22'
}

# =============================================
# Converts a matplotlib figure to a base64
# image string that can be embedded in HTML
# =============================================
def convertGraphToImage():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=130)
    img.seek(0)
    graph = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return graph

# =============================================
# Route "/" → Homepage
# =============================================
@app.route("/")
def index():
    return render_template("index.html")

# =============================================
# Route "/page1" → Researching the Brief
# =============================================
@app.route("/page1")
def page1():
    return render_template("page1.html")

# =============================================
# Route "/page2" → Dataset & Cleaning
# =============================================
@app.route("/page2")
def page2():
    return render_template("page2.html")

# =============================================
# Route "/page3" → Mean, Median, Mode
# =============================================
@app.route('/page3', methods=['GET', 'POST'])
def page3():
    sleep_mean = sleep_median = sleep_mode = None
    happy_mean = happy_median = happy_mode = None
    sleep_interpretation = happy_interpretation = None

    if request.method == 'POST':
        df = pd.read_csv('Mental_Health_Lifestyle_Dataset.csv')

        # Clean the data (same steps as page 2)
        df = df.drop_duplicates()
        df = df.dropna(subset=['Sleep Hours', 'Stress Level', 'Mental Health Condition', 'Happiness Score'])
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df = df[(df['sleep_hours'] >= 0) & (df['sleep_hours'] <= 24)]
        df = df[(df['happiness_score'] >= 0) & (df['happiness_score'] <= 10)]

        sleep_mean   = round(df['sleep_hours'].mean(), 2)
        sleep_median = round(df['sleep_hours'].median(), 2)
        sleep_mode   = round(df['sleep_hours'].mode()[0], 2)

        happy_mean   = round(df['happiness_score'].mean(), 2)
        happy_median = round(df['happiness_score'].median(), 2)
        happy_mode   = round(df['happiness_score'].mode()[0], 2)

        sleep_interpretation = (
            f"The mean sleep of {sleep_mean} hours is slightly below the NHS-recommended 7–9 hours. "
            f"The median of {sleep_median} and mode of {sleep_mode} are close to the mean, "
            f"suggesting the data is fairly evenly distributed with no extreme outliers."
        )
        happy_interpretation = (
            f"The mean happiness score of {happy_mean} out of 10 suggests a moderate level of wellbeing. "
            f"The median of {happy_median} is close to the mean, but the mode of {happy_mode} is notably lower, "
            f"hinting at a cluster of particularly unhappy participants in the dataset."
        )

    return render_template('page3.html',
        sleep_mean=sleep_mean, sleep_median=sleep_median, sleep_mode=sleep_mode,
        happy_mean=happy_mean, happy_median=happy_median, happy_mode=happy_mode,
        sleep_interpretation=sleep_interpretation,
        happy_interpretation=happy_interpretation
    )

# =============================================
# Route "/page4" → Line Graph (Age vs Happiness)
# =============================================
def lineGraph():
    df['age_group'] = pd.cut(df['age'], bins=[17, 25, 35, 45, 55, 65],
                             labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    line_data = df.groupby('age_group', observed=True)['happiness_score'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(line_data['age_group'], line_data['happiness_score'],
            marker='o', linewidth=2.5, color='#4575b4',
            markersize=8, markerfacecolor='white', markeredgewidth=2.5)
    ax.fill_between(range(len(line_data)), line_data['happiness_score'], alpha=0.1, color='#4575b4')
    ax.set_title('Average Happiness Score by Age Group', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Age Group', fontsize=11)
    ax.set_ylabel('Average Happiness Score (0-10)', fontsize=11)
    ax.set_xticks(range(len(line_data)))
    ax.set_xticklabels(line_data['age_group'])
    ax.set_ylim(0, 8)
    for i, val in enumerate(line_data['happiness_score']):
        ax.annotate(f'{val:.2f}', (i, val), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return convertGraphToImage()

@app.route("/page4", methods=["GET", "POST"])
def page4():
    graph = None
    if request.method == "POST":
        graph = lineGraph()
    return render_template("page4.html", graph=graph)

# =============================================
# Route "/page5" → Scatter Graph (Sleep vs Happiness)
# =============================================
def scatterGraph():
    corr, pval = stats.pearsonr(df['sleep_hours'], df['happiness_score'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['sleep_hours'], df['happiness_score'], alpha=0.3, color='#4575b4', s=20)
    m, b = np.polyfit(df['sleep_hours'], df['happiness_score'], 1)
    x_line = np.linspace(df['sleep_hours'].min(), df['sleep_hours'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='#e74c3c', linewidth=2,
            label=f'Trend line (r = {corr:.2f})')
    ax.set_title('Sleep Hours vs Happiness Score', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Sleep Hours', fontsize=11)
    ax.set_ylabel('Happiness Score (0-10)', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return convertGraphToImage()

@app.route("/page5", methods=["GET", "POST"])
def page5():
    graph = None
    if request.method == "POST":
        graph = scatterGraph()
    return render_template("page5.html", graph=graph)

# =============================================
# Route "/page6" → Bar Graph (Condition vs Happiness)
# =============================================
def barGraph():
    order = df.groupby('mental_health_condition')['happiness_score'].mean().sort_values(ascending=False).index
    avg_happy = df.groupby('mental_health_condition')['happiness_score'].mean().reindex(order).reset_index()
    colors = [palette.get(c, '#95a5a6') for c in avg_happy['mental_health_condition']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(avg_happy['mental_health_condition'].str.title(),
                  avg_happy['happiness_score'],
                  color=colors, edgecolor='white', linewidth=1.5, width=0.6)
    ax.set_title('Average Happiness Score by Mental Health Condition', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Mental Health Condition', fontsize=11)
    ax.set_ylabel('Average Happiness Score (0-10)', fontsize=11)
    ax.set_ylim(0, 8)
    for bar, val in zip(bars, avg_happy['happiness_score']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return convertGraphToImage()

@app.route("/page6", methods=["GET", "POST"])
def page6():
    graph = None
    if request.method == "POST":
        graph = barGraph()
    return render_template("page6.html", graph=graph)

# =============================================
# Route "/page7" → Pie Chart (Condition Distribution)
# =============================================
def pieChart():
    condition_counts = df['mental_health_condition'].value_counts()
    colors_pie = [palette.get(c, '#95a5a6') for c in condition_counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        condition_counts.values,
        labels=condition_counts.index.str.title(),
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for text in autotexts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    ax.set_title('Distribution of Mental Health Conditions', fontsize=14, fontweight='bold', pad=15)
    return convertGraphToImage()

@app.route("/page7", methods=["GET", "POST"])
def page7():
    graph = None
    if request.method == "POST":
        graph = pieChart()
    return render_template("page7.html", graph=graph)

# =============================================
# Route "/page8" → Linear Regression
# =============================================
def regressionChart():
    slope, intercept, r, p, se = stats.linregress(df['sleep_hours'], df['happiness_score'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df['sleep_hours'], df['happiness_score'],
               alpha=0.3, color='#4575b4', s=20, label='Data points')
    x_line = np.linspace(df['sleep_hours'].min(), df['sleep_hours'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='#e74c3c', linewidth=2.5,
            label=f'y = {slope:.2f}x + {intercept:.2f}  (R² = {r**2:.4f})')
    ax.set_title('Linear Regression: Sleep Hours vs Happiness Score', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Sleep Hours (Independent Variable)', fontsize=11)
    ax.set_ylabel('Happiness Score (Dependent Variable)', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return convertGraphToImage()

@app.route("/page8", methods=["GET", "POST"])
def page8():
    graph = None
    if request.method == "POST":
        graph = regressionChart()
    return render_template("page8.html", graph=graph)

# =============================================
# Route "/page9" → Conclusion & References
# =============================================
@app.route("/page9")
def page9():
    return render_template("page9.html")

# =============================================
# FLASK SERVER
# =============================================
if __name__ == "__main__":
    app.run(debug=True)
