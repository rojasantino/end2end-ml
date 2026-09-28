import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_student_dataset(n=500, seed=42):
    np.random.seed(seed)

    male_names   = ['Arjun','Karthik','Vijay','Ravi','Suresh','Mohan','Ganesh',
                    'Siva','Bala','Dinesh','Mani','Arun','Kumar','Naveen','Pradeep']
    female_names = ['Priya','Deepa','Kavya','Anjali','Meena','Lakshmi','Pooja',
                    'Nithya','Saranya','Divya','Ananya','Revathi','Bhavani','Keerthana','Sneha']
    last_names   = ['Kumar','Rajan','Selvam','Murugan','Krishnan','Subramanian',
                    'Pandian','Ramesh','Narayanan','Devi','Pillai','Shankar','Raj','Nair','Velu']

    departments    = ['Computer Science','Electronics','Mechanical','Civil','Information Technology']
    regions        = ['Chennai','Coimbatore','Madurai','Trichy','Salem','Erode','Vellore','Tirunelveli']
    parent_edus    = ['No Schooling','School','Diploma','Graduate','Post-Graduate']
    parent_weights = [0.05, 0.20, 0.25, 0.35, 0.15]

    genders     = np.random.choice(['Male', 'Female'], n)
    first_names = [
        np.random.choice(male_names) if g == 'Male' else np.random.choice(female_names)
        for g in genders
    ]

    df = pd.DataFrame({
        'student_id'          : [f'STU{1000+i}' for i in range(n)],
        'name'                : [f'{fn} {np.random.choice(last_names)}' for fn in first_names],
        'gender'              : genders,
        'age'                 : np.random.randint(18, 23, n),
        'department'          : np.random.choice(departments, n),
        'region'              : np.random.choice(regions, n),
        'attendance_pct'      : np.clip(np.random.normal(78, 12, n), 40, 100).round(1),
        'study_hours_per_day' : np.clip(np.random.normal(4.5, 1.5, n), 1, 10).round(1),
        'math_score'          : np.clip(np.random.normal(68, 15, n), 20, 100).round(0).astype(int),
        'science_score'       : np.clip(np.random.normal(65, 14, n), 20, 100).round(0).astype(int),
        'english_score'       : np.clip(np.random.normal(70, 12, n), 25, 100).round(0).astype(int),
        'programming_score'   : np.clip(np.random.normal(63, 18, n), 15, 100).round(0).astype(int),
        'tamil_score'         : np.clip(np.random.normal(72, 11, n), 30, 100).round(0).astype(int),
        'part_time_job'       : np.random.choice(['Yes', 'No'], n, p=[0.25, 0.75]),
        'internet_access'     : np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
        'parent_education'    : np.random.choice(parent_edus, n, p=parent_weights),
        'family_income_lpa'   : np.clip(np.random.exponential(4.5, n), 1, 25).round(1),
    })

    missing_cols = ['math_score', 'science_score', 'attendance_pct', 'family_income_lpa']
    for col in missing_cols:
        idx = np.random.choice(df.index, size=int(n * 0.04), replace=False)
        df.loc[idx, col] = np.nan

    return df


# ── Generate dataset ─────────────────────────────────────────────────────────
df_raw = generate_student_dataset(n=500)
df_raw.to_csv('student_performance.csv', index=False)

print(f'✅ Dataset generated: {df_raw.shape[0]} students × {df_raw.shape[1]} features')
print(f'   Saved to: student_performance.csv')
df_raw.head()

# ── Basic structure ──────────────────────────────────────────────────────────
print('=' * 55)
print(f'  Rows    : {df_raw.shape[0]}')
print(f'  Columns : {df_raw.shape[1]}')
print('=' * 55)
print('\nColumn data types:')
print(df_raw.dtypes)
print('\nBasic statistics (numeric columns):')
df_raw.describe().round(2)

# ── Missing value audit ──────────────────────────────────────────────────────
missing     = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_report = pd.DataFrame({
    'Missing Count' : missing,
    'Missing %'     : missing_pct,
    'Data Type'     : df_raw.dtypes
}).query('`Missing Count` > 0')

print('🔍 Missing Value Report:')
print(missing_report.to_string())
print(f'\n   Total cells  : {df_raw.size:,}')
print(f'   Missing cells: {df_raw.isnull().sum().sum():,}')

dup_count = df_raw.duplicated().sum()
print(f'🔁 Duplicate rows: {dup_count}')

cat_cols = ['gender','department','region','part_time_job','internet_access','parent_education']
for col in cat_cols:
    print(f'\n📌 {col}:')
    print(df_raw[col].value_counts().to_string())


# ── 3. Clean data ────────────────────────────────────────────────────────────
df = df_raw.copy()

# 3.1 Fill missing numeric values with median
numeric_missing = ['math_score','science_score','attendance_pct','family_income_lpa']
for col in numeric_missing:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f' {col:<25} → filled {df_raw[col].isnull().sum()} NaNs with median = {median_val:.2f}')

remaining_nulls = df.isnull().sum().sum()
print(f'\n   Remaining missing values: {remaining_nulls}' if remaining_nulls == 0 else f'   Still {remaining_nulls} missing!')

# 3.2 Fix data types
for col in ['math_score','science_score']:
    df[col] = df[col].astype(int)

cat_cols = ['gender','department','region','part_time_job','internet_access','parent_education']
for col in cat_cols:
    df[col] = df[col].astype('category')

print('Data types after cleaning:')
print(df.dtypes)

# 3.3 Validate score ranges
score_cols = ['math_score','science_score','english_score','programming_score','tamil_score']
print('\n📏 Score range validation:')
for col in score_cols:
    lo, hi = df[col].min(), df[col].max()
    status = '✅' if lo >= 0 and hi <= 100 else '⚠️ OUT OF RANGE'
    print(f'   {col:<22} min={lo:3d}  max={hi:3d}  {status}')


# ── 4. Feature engineering ───────────────────────────────────────────────────
score_cols = ['math_score','science_score','english_score','programming_score','tamil_score']

df['total_score']   = df[score_cols].sum(axis=1)
df['average_score'] = df[score_cols].mean(axis=1).round(2)

def assign_grade(avg):
    if avg >= 85:   return 'A'
    elif avg >= 70: return 'B'
    elif avg >= 55: return 'C'
    elif avg >= 40: return 'D'
    else:           return 'F'

df['grade']  = df['average_score'].apply(assign_grade)
df['result'] = df[score_cols].apply(lambda row: 'Pass' if (row >= 40).all() else 'Fail', axis=1)

q75 = df['average_score'].quantile(0.75)
q25 = df['average_score'].quantile(0.25)

def performance_tier(avg):
    if avg >= q75:   return 'Top'
    elif avg <= q25: return 'Weak'
    else:            return 'Average'

df['performance_tier']    = df['average_score'].apply(performance_tier)
df['study_efficiency']    = (df['average_score'] / df['study_hours_per_day']).round(2)
df['attendance_category'] = pd.cut(
    df['attendance_pct'],
    bins=[0, 60, 75, 85, 100],
    labels=['Poor (<60%)','Average (60–75%)','Good (75–85%)','Excellent (>85%)']
)

print('✅ New features created:')
for col in ['total_score','average_score','grade','result','performance_tier','study_efficiency','attendance_category']:
    print(f'   {col}')

df[['name','average_score','grade','result','performance_tier','study_efficiency']].head(8)


# ── 5. Summary report ────────────────────────────────────────────────────────
print('='*60)
print('            STUDENT PERFORMANCE SUMMARY REPORT')
print('='*60)

total  = len(df)
passed = (df['result'] == 'Pass').sum()
failed = total - passed

print(f'\n  Total Students  : {total}')
print(f'  Passed          : {passed} ({passed/total*100:.1f}%)')
print(f'  Failed          : {failed} ({failed/total*100:.1f}%)')
print(f'  Overall Average : {df["average_score"].mean():.2f} / 100')
print(f'  Highest Average : {df["average_score"].max():.2f}')
print(f'  Lowest Average  : {df["average_score"].min():.2f}')

print('\n--- Grade Distribution ---')
grade_counts = df['grade'].value_counts().sort_index()
for grade, cnt in grade_counts.items():
    bar = '█' * (cnt // 5)
    print(f'  {grade} : {bar} {cnt} students ({cnt/total*100:.1f}%)')

print('\n--- Performance Tier ---')
for tier, cnt in df['performance_tier'].value_counts().items():
    print(f'  {tier:<10}: {cnt} students')

print('\n--- Subject Averages ---')
for col in score_cols:
    subj = col.replace('_score','').title()
    print(f'  {subj:<15}: {df[col].mean():.2f}')

print('\n--- Average Score by Department ---')
dept_stats = df.groupby('department', observed=True)['average_score'].agg(['mean','std','count']).round(2)
dept_stats.columns = ['Mean Score','Std Dev','Students']
print(dept_stats.sort_values('Mean Score', ascending=False).to_string())

print('\n--- Average Score by Gender ---')
print(df.groupby('gender', observed=True)['average_score'].agg(['mean','count']).round(2).to_string())

print('\n--- Does Part-time Job Affect Score? ---')
print(df.groupby('part_time_job', observed=True)['average_score'].agg(['mean','count']).round(2).to_string())

corr = df['study_hours_per_day'].corr(df['average_score'])
print(f'\n--- Correlation: Study Hours vs Average Score ---')
print(f'   Pearson r = {corr:.3f}  ({"positive" if corr > 0 else "negative"} relationship)')


# ── 6. Dashboard ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Student Performance Dashboard', fontsize=16, fontweight='bold', y=1.01)

colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']

# Plot 1: Grade distribution
grade_counts = df['grade'].value_counts().sort_index()
axes[0,0].bar(grade_counts.index, grade_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0,0].set_title('Grade Distribution')
axes[0,0].set_xlabel('Grade')
axes[0,0].set_ylabel('Number of Students')
for i, (g, v) in enumerate(grade_counts.items()):
    axes[0,0].text(i, v+2, str(v), ha='center', fontweight='bold')

# Plot 2: Score distribution (histogram)
axes[0,1].hist(df['average_score'], bins=20, color='#3498db', edgecolor='white', linewidth=0.8, alpha=0.85)
axes[0,1].axvline(df['average_score'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {df["average_score"].mean():.1f}')
axes[0,1].set_title('Average Score Distribution')
axes[0,1].set_xlabel('Average Score')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()

# Plot 3: Subject-wise box plot  ← FIXED
score_data = df[score_cols].rename(columns={
    'math_score'       : 'Math',
    'science_score'    : 'Science',
    'english_score'    : 'English',
    'programming_score': 'Prog.',
    'tamil_score'      : 'Tamil'
})
df_scores = df[score_cols].copy()
df_scores.columns = ['Math','Science','English','Prog.','Tamil']
df_scores.plot(kind='box', ax=axes[0,2], patch_artist=True,
               boxprops=dict(facecolor='#ecf0f1'), medianprops=dict(color='red', linewidth=2))
axes[0,2].set_title('Score Distribution by Subject')
axes[0,2].set_ylabel('Score')

# Plot 4: Department vs average score
dept_avg = df.groupby('department', observed=True)['average_score'].mean().sort_values()
axes[1,0].barh(dept_avg.index, dept_avg.values, color='#9b59b6', edgecolor='white')
axes[1,0].set_title('Avg Score by Department')
axes[1,0].set_xlabel('Average Score')
for i, v in enumerate(dept_avg.values):
    axes[1,0].text(v+0.3, i, f'{v:.1f}', va='center')

# Plot 5: Study hours vs Average score (scatter)
axes[1,1].scatter(df['study_hours_per_day'], df['average_score'],
                  alpha=0.35, s=25, c='#e74c3c', edgecolors='none')
z = np.polyfit(df['study_hours_per_day'], df['average_score'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['study_hours_per_day'].min(), df['study_hours_per_day'].max(), 100)
axes[1,1].plot(x_line, p(x_line), 'b--', linewidth=2, label=f'r = {corr:.2f}')
axes[1,1].set_title('Study Hours vs Average Score')
axes[1,1].set_xlabel('Study Hours / Day')
axes[1,1].set_ylabel('Average Score')
axes[1,1].legend()

# Plot 6: Pass / Fail by part-time job
pj_result = df.groupby(['part_time_job','result'], observed=True).size().unstack(fill_value=0)
pj_result.plot(kind='bar', ax=axes[1,2], color=['#e74c3c','#2ecc71'], edgecolor='white')
axes[1,2].set_title('Pass/Fail by Part-time Job')
axes[1,2].set_xlabel('Has Part-time Job?')
axes[1,2].set_ylabel('Count')
axes[1,2].tick_params(axis='x', rotation=0)
axes[1,2].legend(title='Result')

plt.tight_layout()
plt.savefig('student_performance_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Dashboard saved as student_performance_dashboard.png')


# ── Save cleaned CSV ─────────────────────────────────────────────────────────
df.to_csv('student_performance_cleaned.csv', index=False)
print('✅ Cleaned dataset saved: student_performance_cleaned.csv')
print(f'   Shape: {df.shape}')

# ── Generate text summary report ─────────────────────────────────────────────
report_lines = [
    '=' * 60,
    '       STUDENT PERFORMANCE ANALYSIS REPORT',
    '=' * 60,
    f'\nTotal Students Analyzed : {len(df)}',
    f'Pass Rate               : {(df["result"]=="Pass").mean()*100:.1f}%',
    f'Overall Average Score   : {df["average_score"].mean():.2f} / 100',
    f'Strongest Subject       : {df[score_cols].mean().idxmax().replace("_score","").title()}',
    f'Weakest Subject         : {df[score_cols].mean().idxmin().replace("_score","").title()}',
    '',
    '--- KEY INSIGHTS ---',
    f'1. {(df["result"]=="Pass").sum()} students passed ({(df["result"]=="Pass").mean()*100:.1f}%). Focus needed on the {(df["result"]=="Fail").sum()} who failed.', #observed=True- Finance    NaN   (no data but still shown)
    f'2. {df.groupby("department", observed=True)["average_score"].mean().idxmax()} dept has the highest avg score ({df.groupby("department", observed=True)["average_score"].mean().max():.1f}).',
    f'3. Students with part-time jobs score {df.groupby("part_time_job", observed=True)["average_score"].mean().diff().iloc[-1]:.1f} pts lower on average.',
    f'4. Study hours and score have a correlation of {corr:.2f} — more study hours = higher scores.',
    f'5. {(df["attendance_pct"] < 75).sum()} students ({(df["attendance_pct"]<75).mean()*100:.1f}%) are below the 75% attendance threshold.',
    '',
    '--- GRADE DISTRIBUTION ---',
] + [f'  Grade {g}: {(df["grade"]==g).sum()} students ({(df["grade"]==g).mean()*100:.1f}%)' for g in ['A','B','C','D','F']]

report_text = '\n'.join(report_lines)

with open('student_performance_report.txt', 'w') as f:
    f.write(report_text)

print('\n✅ Summary report saved: student_performance_report.txt')
print('\n' + report_text)