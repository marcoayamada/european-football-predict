from sklearn.externals import joblib
import pickle
import pandas as pd

classifier = joblib.load(r'modelos/190123-xgbmodel.save')
scaler = pickle.load(open(r'transformacoes/190123-minmax.save','rb'))

def make_pred(body):
	# formato body	#{'lista_jogos': [{'away_pos': 10, 'elo_away_score': 1463.85192871, 'elo_home_score': 1426.95825195, 'fifa_away_att': 71, 'fifa_away_def': 71, 'fifa_away_mid': 70, 'fifa_away_ova': 70, 'fifa_home_att': 70, 'fifa_home_def': 69, 'fifa_home_mid': 68, 'fifa_home_ova': 69, 'home_pos': 8, 'last5all_home_away_dif': 7, 'tfm_value_away': 21600000, 'tfm_value_home': 13450000}]}
	game_data = pd.DataFrame(body['lista_jogos']) #[{'away_pos': 10, 'elo_away_score': 1463.85192871, 'elo_home_score': 1426.95825195, 'fifa_away_att': 71, 'fifa_away_def': 71, 'fifa_away_mid': 70, 'fifa_away_ova': 70, 'fifa_home_att': 70, 'fifa_home_def': 69, 'fifa_home_mid': 68, 'fifa_home_ova': 69, 'home_pos': 8, 'last5all_home_away_dif': 7, 'tfm_value_away': 21600000, 'tfm_value_home': 13450000}]}
	x_scaled = scaler.transform(game_data)
	preds = pd.DataFrame(classifier.predict_proba(x_scaled), columns=classifier.classes_).to_json(orient='records')
	# jogando resultados em um formato json valido
	result={}
	result['preds'] = preds

	return result