import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Loads the data from the given messages and categories file path.

    Args:
        messages_filepath (str): File path of the messages to load.
        categories_filepath (str): File path of the categories to load.
    Returns:
        df (pd.DataFrame): Dataframe of the merged Messages and Categories files data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id')
    return df


def process_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process, clean and merge the categories dataset into the main one.

    Args:
        df (pd.DataFrame): Dataframe of the merged Messages and Categories files

    Returns:
        df (pd.DataFrame): Dataframe cleaned and processed by categories
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: 1 if '1' in x else 0)

    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function responsible for cleaning the data.

    Args:
        df (pd.DataFrame): DataFrame to be cleaned.
    Returns:
        df_cleaned (pd.DataFrame): Cleaned DataFrame.
    """
    df_cleaned = process_categories(df)
    df_cleaned.drop_duplicates(inplace=True)

    return df_cleaned


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Save the Dataframe passed to a sqlite database.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        database_filename (str): Filename of the database.
    """
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')


def main():
    """
    Main function responsible to handle all the ETL pipeline process.
    Extract:
        load_data()
    Transform:
        clean_data()
    Load:
        save_data()
    """
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
