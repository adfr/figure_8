import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
      '''
    Load the data from CVS files
    Input:
      -Path to the csv file of messages
      -Path to the csv file of categories
    output:
      - a merged data frame
    '''
  messages = pd.read_csv(messages_filepath)
  categories = pd.read_csv(categories_filepath)
  return  messages.merge(categories, how='inner' , left_on='id' , right_on='id')


def clean_data(df):
   '''
    Clean the data
    Input:
      -Dataframe
    output:
      - cleaned data frame: split the category per column, remove duplicates, 
    '''
  categories = df.categories.str.split(pat=';', n=-1, expand=True)
  row = categories.loc[0,:]
  category_colnames = row.apply( lambda col : col[0:len(col)-2])
  categories.columns = category_colnames
  for column in categories:
      # set each value to be the last character of the string
      categories[column] = categories[column].astype(str).apply( lambda row : row[len(row)-1] )
      
      # convert column from string to numeric
      categories[column] = categories[column].astype(int)
  df = pd.concat([df,categories] , axis =1)
  df=df.drop(labels = 'categories' , axis =1)
  df = df.drop_duplicates(subset=['id'], keep='first', inplace=False)
  df = df.drop_duplicates(subset=['message'], keep='first', inplace=False)
  return df

def save_data(df, database_filename):
  engine = create_engine('sqlite:///'+database_filename)
  df.to_sql('Message_label', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()