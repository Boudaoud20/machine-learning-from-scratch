import numpy as np
import pandas as pd

class OrdinalEncoder:
    def __init__(self, categories:list[str]):
        self.categories: list[str]= categories
        self.map_: dict[str, int] = {}
        self.is_fitted: bool = False

    @property
    def categories(self)->list[str]:
        return self._categories

    @categories.setter
    def categories(self, categories:list[str])->None:
        if not isinstance(categories,list):
            raise TypeError("You need to pass list of ordinal strings.")
        if not all(isinstance(category, str) for category in categories):
            raise TypeError("All categories must be strings.")
        if not categories:
            raise ValueError("categories must not be empty.")
        if len(categories) != len(set(categories)):
            raise ValueError("Categories must be unique.")
        self._categories = categories

    @property
    def is_fitted(self)->bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self,is_fitted)->None:
        if not isinstance(is_fitted,bool):
            raise TypeError("You need to pass boolean value.")
        self._is_fitted = is_fitted


    @property
    def map_(self)-> dict[str, int]:
        return self._map_

    @map_.setter
    def map_(self, map_:dict[str, int])->None:
        if not isinstance(map_, dict):
            raise TypeError("You need to pass a dict.")
        self._map_ = map_



    def fit(self) -> None:
        """
        This method creates a mapping between each category and its corresponding integer value.

        :return: none
        :rtype: None
        """
        map_= {category:i  for i,category  in enumerate(self.categories)}
        self.map_=map_
        self.is_fitted = True

    def transform(self, input:pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        This method replaces each category in the input data with its corresponding integer value.

        :param input: A pandas Series or DataFrame containing the categories to encode.
        :type input: pd.Series | pd.DataFrame

        :raises ValueError: If the encoder has not been fitted, the mapping is empty, or unknown categories are found.
        :raises TypeError: If the input is not a pandas Series or DataFrame.

        :returns: A pandas Series or DataFrame containing the encoded values.
        :rtype: pd.Series | pd.DataFrame
        """
        if not self.is_fitted:
            raise ValueError("You didn't fit the encoder yet.")
        if not self.map_:
            raise ValueError("The passed dictionary is empty.")
        if not isinstance(input, (pd.Series, pd.DataFrame)):
            raise TypeError("Input must be a pandas Series or DataFrame.")
        
        if isinstance(input, pd.Series):
            unknown = input[~input.isin(self.map_)]
        elif isinstance(input, pd.DataFrame):
            unknown = input[~input.isin(self.map_)].stack()
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")
        
        if not unknown.empty:
            raise ValueError(f"Unknown categories found: {unknown.unique().tolist()}")
        
        output: pd.Series | pd.DataFrame = input.replace(self.map_)
        return output

    def fit_transform(self, input:pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        This method fits the encoder and transforms the input data.


        :param input: A pandas Series or DataFrame containing the categories to encode.
        :type input: pd.Series | pd.DataFrame

        :returns: A pandas Series or DataFrame containing the encoded values.
        :rtype: pd.Series | pd.DataFrame
        """
        self.fit()
        return self.transform(input=input)


    
        

class OneHotEncoder:

    def __init__(self):
        self.is_fitted: bool = False
        self.map_: dict[str, list[int]] = {}

    @property
    def is_fitted(self)->bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self,is_fitted)->None:
        if not isinstance(is_fitted,bool):
            raise TypeError("You need to pass boolean value.")
        self._is_fitted = is_fitted


    def fit(self, input:pd.Series | pd.DataFrame) -> None:
        """
        This method identifies the unique categories in the input data and creates a binary vector for each category.

        :param input: A pandas Series or DataFrame containing the categories to encode.
        :type input: pd.Series | pd.DataFrame

        :raises ValueError: If the input is not a pandas Series or DataFrame.

        :returns: None
        :rtype: None
        """
        self.map_ = {}
        if isinstance(input, pd.Series):
            v = set()
            for element in input.to_numpy():
                v.add(element)
            length = len(v)
            for index, e in enumerate(sorted(v)):
                vector = np.zeros(length, dtype=int)
                vector[index] = 1
                self.map_[e] = vector
            
        elif isinstance(input, pd.DataFrame):
            column_n = len(input.columns)
            for i in range(column_n):
                v = set()
                for element in input.iloc[::,i].to_numpy():
                    v.add(element)
                length = len(v)
                result = {}
                for index, e in enumerate(sorted(v)):
                    vector = np.zeros(length, dtype=int)
                    vector[index] = 1
                    result[e] = vector
                self.map_[input.columns[i]] = result
        else:
            raise ValueError("You need to pass Series or DataFrame.")
        self.is_fitted = True
    
    
    def transform(self, input: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        This method replaces each category with its corresponding binary vector.

        :param input: A pandas Series or DataFrame containing the categories to encode.
        :type input: pd.Series | pd.DataFrame

        :raises ValueError: If the encoder has not been fitted or the mapping is empty.

        :returns: A pandas Series or DataFrame containing the encoded values.
        :rtype: pd.Series | pd.DataFrame
        """
        if not self.is_fitted:
            raise ValueError("You didn't fit the encoder yet.")
        if not self.map_:
            raise ValueError("The passed dictionary is empty.")

        output = input.copy()
        if isinstance(output, pd.Series):
            return output.map(self.map_)
        for key, value in self.map_.items():
            output[key] = output[key].map(value)
        return output

    def fit_transform(self, input:pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        This method fits the encoder and transforms the input data.

        :param input: A pandas Series or DataFrame containing the categories to encode.
        :type input: pd.Series | pd.DataFrame

        :returns: A pandas Series or DataFrame containing the encoded values.
        :rtype: pd.Series | pd.DataFrame
        """
        self.fit(input=input)
        return self.transform(input=input)



class BinaryEncoder:
    """
    I will fill it later, this approach will be good for memory efficiency.
    np.ceil(  log2(categories number)   )
    """