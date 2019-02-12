# Import useful packages
import pandas as pd
import numpy as np
import csv
import pickle
import statsmodels.api as sm

# Generate df_expanded that retains all labels in categorical variables (i.e., it is more nuanced) than file df created in IPython Notebook #1.

def create_dummies_fn(df):

	# Drop "In progress" observations
	df = df[df['Company response to consumer'] != 'In progress']
	# Select only credit card-related complaints
	df = df[df.Product == 'Credit card']
	# Generate a binary outcome variable - closed with monetary relief (1) or otherwise (0)
	df['Y'] = np.where(df['Company response to consumer'] == 'Closed with monetary relief', 1,0)


	print("--- # of categories in variable 'Issue': ", df.Issue.nunique())
	print()
	# Create indicator / dummy variables for categories of 'Issue'
	# Arbitrarily, let 'Other' be the excluded category in regressions (i.e., drop it) 
	df_Issue = pd.get_dummies(df['Issue'], prefix='Issue') 
	df_Issue = df_Issue.drop(['Issue_Other'], axis=1)
	df = pd.concat([df, df_Issue], axis=1)


	print("--- 'Consumer complaint narrative' is either NUll or contains text of a complaint ")
	print()
	# Create an indicator variable for the presense of a complaint narrative 
	df['Is_narrative'] = np.where(df['Consumer complaint narrative'].notnull(), 1, 0)


	print('--- # of companies: ', df.Company.nunique())
	print()
	# Create Company dummy variables
	# Arbitrarily, let 'U.S. Bancorp' be the excluded category in regressions (i.e., drop it)
	df_Company = pd.get_dummies(df['Company'], prefix='Company')
	df_Company = df_Company.drop(['Company_U.S. Bancorp'], axis=1)
	df = pd.concat([df, df_Company], axis=1)


	print('--- # of U.S. states and territories: ',df.State.nunique())
	print()
	# Create dummy variables for categories in the State variable
	# Make Nulls the excluded (baseline) category in regressions
	df_States = pd.get_dummies(df['State'], prefix="State") 
	df = pd.concat([df, df_States], axis=1)


	print('--- # of Tags: ',df.Tags.nunique())
	print()
	# Create dummy variables for categories in the Tags variable
	# Make Nulls the excluded category
	df_Tags = pd.get_dummies(df['Tags'], prefix='Tags') 
	df = pd.concat([df, df_Tags], axis=1)


	print("--- # of categories in 'Consumer consent provided?': ",df['Consumer consent provided?'].nunique())
	print()
	# Create dummy variables for categories in Variable "Consumer consent provided?"
	# Make Nulls the excluded category
	df_consent = pd.get_dummies(df['Consumer consent provided?'], prefix='Consent') 
	df = pd.concat([df, df_consent], axis=1)

	print('--- # of Submission categories: ',df['Submitted via'].nunique())
	print()
	# Create dummy variables for categories in Variable "Submitted via"
	# Make Postal Mail the excluded category
	df_Submitted = pd.get_dummies(df['Submitted via'], prefix='Submitted') 
	df_Submitted = df_Submitted.drop(['Submitted_Postal mail'], axis=1)
	df = pd.concat([df, df_Submitted], axis=1)


	print("--- Binary variable 'Timely_response'")
	print()
	# Create an indicator variable for the presense of a complaint narrative 
	df['Timely_response'] = np.where(df['Timely response?'] == 'No', 1, 0)


	print("--- Varibale Year")
	print()
	# Create Year variable from Date received. Do not drop year 2011
	df['Year'] = df['Date received'].str[6:]
	# Create yearly dummies and make 2012 the excluded category
	df_Year = pd.get_dummies(df['Year'], prefix="Year")
	df_Year = df_Year.drop(['Year_2012'], axis =1)
	df = pd.concat([df, df_Year], axis=1)
    

	print("--- Variable Month")
	print()
	# Create Month variable from Date received
	df['Month'] = df['Date received'].str[:2]
	# Create monthly dummy variables. Let April be the baseline (excluded) category
	df_Months = pd.get_dummies(df['Month'], prefix='Month') 
	df_Months = df_Months.drop(['Month_04'], axis=1)
	df = pd.concat([df, df_Months], axis=1)

	# Add a constant term to df
	print('--- Add constant term')
	df = sm.add_constant(df)
	print()
    
	print('Done creating dummy variables')
	print('______________________________________')
	return df


def add_ACS_data_fn(df, ACS_data, census_vars):

	# Take natural log of ACS's household median income
	ACS_data['ln_median_income'] = np.log(ACS_data.Median_household_income)
    
	#American Community Survey (Census) Data
	# Remove empty space from variable ZIP code
	ACS_data["ZIP code"] = ACS_data["ZIP code"].str.strip()

	# CFPB data
	# Remove empty space from variable ZIP code
	df["ZIP code"] = df["ZIP code"].str.strip()

	# Select a subset of ACS variables
	ACS_data = ACS_data[census_vars]

	# Add ACS data to the CFPB data
	df = df.merge(ACS_data, on = 'ZIP code', how = 'left', indicator = True)

	# Replace missing ACS data with means #note: 1st var in census_vars is 'ZIP code'
	for i in census_vars[1:]:
		df[i] = df[i].fillna(df[i].mean())

    # Create an indicator variable to mark Zip codes in the CFPB data that could not be matched with ACS data 
	# (missing values)
	df['ACS_missing'] = np.where(df._merge == 'left_only', 1, 0)
	df = df.drop(['_merge'], axis=1)
    
	print('Done adding ACS (Census data)')
	print('______________________________________')
	return df


def zip_code_dummies(df):	
	#Create 3-digit zip codes
	df['3digit_zip'] = df['ZIP code'].str[:3]     
	#Create 3-digit zip Code dummies
	df_zip = pd.get_dummies(df['3digit_zip'], prefix='ZIP') 
	df = pd.concat([df, df_zip], axis=1)
	df = df.drop(['3digit_zip'], axis=1)    
	print("# of 3-digit zip codes: ", df_zip.shape[1])
	print()
	print('Done creating 3-digit ZIP code dummies')
	print('______________________________________')
	return df

def drop_columns_for_logit(df):

	# Drop redundant variables
	columns_to_drop = ['Sub-product', 'Sub-issue', 'Company', 'Issue', 'Complaint ID', 'Product', 
					'Date sent to company', 'Consumer complaint narrative', 'State', 'Tags',
					'Consumer consent provided?', 'Submitted via', 'Date received', 'Year', 'Month',
					'ZIP code', 'Timely response?', 'Company response to consumer', 'Company public response',
					'Consumer disputed?']
	df = df.drop(columns = columns_to_drop, axis =1)
	print('Done dropping redundant columns')
	print('______________________________________')
	return df
