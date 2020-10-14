#This file enables populating dividends.sqlite
import sqlite3
import json
import numpy as np
import os

def run_sqlite(scriptDirectory):
	sql_create_projects_table = ''' CREATE TABLE IF NOT EXISTS repo_schedules 
									 (indice text, t1 real, t2 real, t3 real, t4 real, t5 real, t6 real, t7 real, t8 real, t9 real, t10 real, t11 real, t12 real, t13 real, t14 real, date text); '''

	sql_insert_repo = ''' INSERT INTO repo_schedules(indice,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,date) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''


	conn = sqlite3.connect(os.path.join(scriptDirectory, 'output', 'repo.sqlite'))
	c = conn.cursor()
	c.execute(sql_create_projects_table)

	#inserting data in the table 
	universe = None
	with open(os.path.join(scriptDirectory, 'output/universe_repo_cleaned.json'), 'r') as f:
		universe = json.load(f)
	values = np.array(list(universe.values()))[:,1]
	dates = np.array(list(universe.values()))[:,2]
	dates = dates.reshape((len(dates),1))    
	indexes = np.array(list(universe.keys()))
	indexes = indexes.reshape((len(indexes),1))
	repo_schedules = np.column_stack((indexes,values,dates))

	for i,div_schedule in enumerate(repo_schedules):
		indice = div_schedule[0]
		date = div_schedule[2]
		#SHANGHAI SE 50 is deleted
		if(indice!="SHANGHAI SE 50"):
			T1 = div_schedule[1][0]
			T2 = div_schedule[1][1]
			T3 = div_schedule[1][2]
			T4 = div_schedule[1][3]
			T5 = div_schedule[1][4]
			T6 = div_schedule[1][5]
			T7 = div_schedule[1][6]
			T8 = div_schedule[1][7]
			T9 = div_schedule[1][8]
			T10 = div_schedule[1][9]
			T11 = div_schedule[1][10]
			T12 = div_schedule[1][11]
			T13 = div_schedule[1][12]
			T14 = div_schedule[1][13]

			c.execute(sql_insert_repo,(indice,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,date))

	conn.commit()
	conn.close()