from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
import os
from django.conf import settings


def load_university_data(data_type='current'):
    """Load university data from CSV file"""
    if data_type == 'projected':
        csv_path = os.path.join(settings.BASE_DIR, 'projected_scores.csv')
    else:
        csv_path = os.path.join(settings.BASE_DIR, 'qs_2025_2026.csv')
    
    df = pd.read_csv(csv_path)
    return df


def find_ku_data(df, data_type='current'):
    """Find Khalifa University data in the dataframe"""
    if data_type == 'projected':
        # In projected scores, the name is in lowercase
        ku_data = df[df['Institution Name'].str.contains('khalifa university', case=False, na=False)]
    else:
        ku_data = df[df['Institution Name'] == 'Khalifa University of Science and Technology']
    
    if ku_data.empty:
        return None
    return ku_data.iloc[0]


def home(request):
    """Main view for the ranking analyzer"""
    # Load both datasets
    current_df = load_university_data('current')
    projected_df = load_university_data('projected')
    
    # Find Khalifa University data in both datasets
    ku_current = find_ku_data(current_df, 'current')
    ku_projected = find_ku_data(projected_df, 'projected')
    
    # Get current ranking and scores
    current_rank = int(ku_current['RANK_2026']) if ku_current is not None else 0
    
    # Create score data for current dataset
    current_score_data = []
    projected_score_data = []
    
    if ku_current is not None:
        current_score_data = [
            {'key': 'AR', 'value': ku_current['SCORE_AR'], 'description': 'Academic Reputation'},
            {'key': 'ER', 'value': ku_current['SCORE_ER'], 'description': 'Employer Reputation'},
            {'key': 'FS', 'value': ku_current['SCORE_FS'], 'description': 'Faculty Student Ratio'},
            {'key': 'CPF', 'value': ku_current['SCORE_CPF'], 'description': 'Citations per Faculty'},
            {'key': 'IF', 'value': ku_current['SCORE_IF'], 'description': 'International Faculty'},
            {'key': 'IS', 'value': ku_current['SCORE_IS'], 'description': 'International Students'},
            {'key': 'ISD', 'value': ku_current['SCORE_ISD'], 'description': 'International Student Diversity'},
            {'key': 'IRN', 'value': ku_current['SCORE_IRN'], 'description': 'International Research Network'},
            {'key': 'EO', 'value': ku_current['SCORE_EO'], 'description': 'Employment Outcomes'},
            {'key': 'ST', 'value': ku_current['SCORE_ST'], 'description': 'Sustainability'},
        ]
    
    # Create score data for projected dataset
    if ku_projected is not None:
        projected_score_data = [
            {'key': 'AR', 'value': ku_projected['SCORE_AR'], 'description': 'Academic Reputation'},
            {'key': 'ER', 'value': ku_projected['SCORE_ER'], 'description': 'Employer Reputation'},
            {'key': 'FS', 'value': ku_projected['SCORE_FS'], 'description': 'Faculty Student Ratio'},
            {'key': 'CPF', 'value': ku_projected['SCORE_CPF'], 'description': 'Citations per Faculty'},
            {'key': 'IF', 'value': ku_projected['SCORE_IF'], 'description': 'International Faculty'},
            {'key': 'IS', 'value': ku_projected['SCORE_IS'], 'description': 'International Students'},
            # {'key': 'ISD', 'value': ku_projected['SCORE_ISD'], 'description': 'International Student Diversity'},
            {'key': 'IRN', 'value': ku_projected['SCORE_IRN'], 'description': 'International Research Network'},
            {'key': 'EO', 'value': ku_projected['SCORE_EO'], 'description': 'Employment Outcomes'},
            {'key': 'ST', 'value': ku_projected['SCORE_ST'], 'description': 'Sustainability'},
        ]
    
    # Current scores dict for JavaScript
    current_scores = {item['key']: item['value'] for item in current_score_data} if current_score_data else {}
    projected_scores = {item['key']: item['value'] for item in projected_score_data} if projected_score_data else {}
    
    context = {
        'current_rank': current_rank,
        'current_score_data': current_score_data,
        'projected_score_data': projected_score_data,
        'current_scores': current_scores,
        'projected_scores': projected_scores,
        'current_scores_json': json.dumps(current_scores) if current_scores else '{}',
        'projected_scores_json': json.dumps(projected_scores) if projected_scores else '{}',
        'ku_current': ku_current,
        'ku_projected': ku_projected,
        'has_projected_data': ku_projected is not None,
    }
    
    return render(request, 'ranking_analyzer/home.html', context)


@csrf_exempt
def calculate_ranking(request):
    """Calculate new ranking based on modified scores"""
    if request.method == 'POST':
        data = json.loads(request.body)
        new_scores = data.get('scores', {})
        data_type = data.get('data_type', 'current')  # 'current' or 'projected'
        
        # Load appropriate data
        df = load_university_data(data_type)
        
        # Create a copy of the dataframe for calculations
        df_calc = df.copy()
        
        # Find and update KU's scores with new values
        ku_data = find_ku_data(df_calc, data_type)
        if ku_data is None:
            return JsonResponse({'error': 'Khalifa University not found in dataset'})
        
        if data_type == 'projected':
            ku_index = df_calc[df_calc['Institution Name'].str.contains('khalifa university', case=False, na=False)].index[0]
        else:
            ku_index = df_calc[df_calc['Institution Name'] == 'Khalifa University of Science and Technology'].index[0]
        
        for score_key, score_value in new_scores.items():
            column_name = f'SCORE_{score_key}'
            df_calc.loc[ku_index, column_name] = float(score_value)
        
        # Calculate new overall score (simplified weighted average)
        # Note: This is a simplified calculation. The actual QS ranking uses complex weightings
        score_weights = {
            'SCORE_AR': 0.3,  # Academic Reputation - 30%
            'SCORE_ER': 0.15,  # Employer Reputation - 15%
            'SCORE_FS': 0.10,  # Faculty Student Ratio - 10%
            'SCORE_CPF': 0.2,  # Citations per Faculty - 20%
            'SCORE_IF': 0.05,  # International Faculty - 5%
            'SCORE_IS': 0.05,  # International Students - 5%
            # 'SCORE_ISD': 0.00,  # International Student Diversity - 0%
            'SCORE_IRN': 0.05,  # International Research Network - 5%
            'SCORE_EO': 0.05,  # Employment Outcomes - 5%
            'SCORE_ST': 0.05,  # Sustainability - 5%
        }
        
        # Calculate new overall scores for all universities
        normalizer = 1.0
        for index, row in df_calc.iterrows():
            weighted_score = 0
            for score_col, weight in score_weights.items():
                if pd.notna(row[score_col]):
                    weighted_score += row[score_col] * weight
            df_calc.loc[index, 'Calculated_Overall_Score'] = weighted_score * 100
        normalizer = df_calc['Calculated_Overall_Score'].max()
        df_calc['Calculated_Overall_Score'] = df_calc['Calculated_Overall_Score'] / normalizer * 100      

        # Sort by calculated overall score (descending) and assign new ranks
        df_calc = df_calc.sort_values('Calculated_Overall_Score', ascending=False)
        df_calc['New_Rank'] = range(1, len(df_calc) + 1)
        
        # Get KU's new rank
        if data_type == 'projected':
            ku_new_data = df_calc[df_calc['Institution Name'].str.contains('khalifa university', case=False, na=False)].iloc[0]
        else:
            ku_new_data = df_calc[df_calc['Institution Name'] == 'Khalifa University of Science and Technology'].iloc[0]
        
        new_rank = int(ku_new_data['New_Rank'])
        new_overall_score = ku_new_data['Calculated_Overall_Score']
        
        # Get surrounding universities (±5 ranks)
        start_rank = max(1, new_rank - 5)
        end_rank = min(len(df_calc), new_rank + 5)
        
        surrounding_unis = df_calc[(df_calc['New_Rank'] >= start_rank) & 
                                 (df_calc['New_Rank'] <= end_rank)][
            ['New_Rank', 'Institution Name', 'Calculated_Overall_Score']
        ].to_dict('records')
        
        # Add country information if available
        if data_type == 'current':
            surrounding_unis_full = df_calc[(df_calc['New_Rank'] >= start_rank) & 
                                          (df_calc['New_Rank'] <= end_rank)][
                ['New_Rank', 'Institution Name', 'Country', 'Calculated_Overall_Score']
            ].to_dict('records')
            surrounding_unis = surrounding_unis_full
        
        # Calculate rank change (only available for current data)
        rank_change = 0
        if data_type == 'current':
            original_rank = int(ku_data['RANK_2026']) if pd.notna(ku_data['RANK_2026']) else new_rank
            rank_change = original_rank - new_rank
        
        return JsonResponse({
            'new_rank': new_rank,
            'new_overall_score': round(new_overall_score, 2),
            'surrounding_universities': surrounding_unis,
            'rank_change': rank_change,
            'data_type': data_type,
        })
    
    return JsonResponse({'error': 'Invalid request method'})
