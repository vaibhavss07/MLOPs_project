import boto3
import os, sys
from src.logger import logging
from src.exception import CustomException
from src.config import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, REGION_NAME
import pandas as pd

from io import StringIO
from typing import Union, List
from mypy_boto3_s3.service_resource import Bucket
from botocore.exceptions import ClientError
import pickle


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, REGION_NAME):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """

        if S3Client.s3_resource==None or S3Client.s3_client==None:

            if AWS_ACCESS_KEY is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY} is not set.")
            
            if AWS_SECRET_ACCESS_KEY is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY} is not set.")
        
            S3Client.s3_resource = boto3.resource('s3',
                                            aws_access_key_id = AWS_ACCESS_KEY,
                                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                            region_name=REGION_NAME
                                            )
            
            S3Client.s3_client = boto3.client('s3',
                                        aws_access_key_id = AWS_ACCESS_KEY,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=REGION_NAME
                                        )
            
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client

    
    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Retrieves the S3 bucket object based on the provided bucket name.

        Args:
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Bucket: S3 bucket object.
        """

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket
        
        except Exception as e:
            raise CustomException(e, sys) 
        
    
    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket.

        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): Key path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Reads the specified S3 object with optional decoding and formatting.

        Args:
            object_name (str): The S3 object name.
            decode (bool): Whether to decode the object content as a string.
            make_readable (bool): Whether to convert content to StringIO for DataFrame usage.

        Returns:
            Union[StringIO, str]: The content of the object, as a StringIO or decoded string.
        """

        try:
            # decode the object content if decode=True else just read the object
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode else object_name.get()["Body"].read()
            )
            # Convert to StringIO if make_readable=True else just return function
            conv_func = lambda: StringIO(func()) if make_readable else func() 
            return conv_func()
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Retrieves the file object(s) from the specified bucket based on the filename.

        Args:
            filename (str): The name of the file to retrieve.
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Union[List[object], object]: The S3 file object or list of file objects.
        """
       
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            return file_objs
        
        except Exception as e:
            raise CustomException(e, sys) 
    

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Loads a serialized model from the specified S3 bucket.

        Args:
            model_name (str): Name of the model file in the bucket.
            bucket_name (str): Name of the S3 bucket.
            model_dir (str): Directory path within the bucket.

        Returns:
            object: The deserialized model object.
        """
        try:
            model_file = model_dir + "/" + model_name if model_dir else model_name
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            logging.info("Production model loaded from S3 bucket.")
            return model
        
        except Exception as e:
            raise CustomException(e, sys) 
    

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Creates a folder in the specified S3 bucket.

        Args:
            folder_name (str): Name of the folder to create.
            bucket_name (str): Name of the S3 bucket.
        """
        try:
            # Check if folder exists by attempting to load it
            self.s3_resource.Object(bucket_name, folder_name).load()
            
        except ClientError as e:
            # If folder does not exist, create it
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj) 
    

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        """
        Uploads a local file to the specified S3 bucket with an optional file deletion.

        Args:
            from_filename (str): Path of the local file.
            to_filename (str): Target file path in the bucket.
            bucket_name (str): Name of the S3 bucket.
            remove (bool): If True, deletes the local file after upload.
        """
        
        try:
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Uploaded {from_filename} to {to_filename} in {bucket_name}")

            # Delete the local file if remove is True
            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file {from_filename} after upload")
          
        except Exception as e:
            raise CustomException(e, sys) 
    

    def upload_df_as_csv(self, data_frame: pd.DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        """
        Uploads a DataFrame as a CSV file to the specified S3 bucket.

        Args:
            data_frame (DataFrame): DataFrame to be uploaded.
            local_filename (str): Temporary local filename for the DataFrame.
            bucket_filename (str): Target filename in the bucket.
            bucket_name (str): Name of the S3 bucket.
        """
        
        try:
            # Save DataFrame to CSV locally and then upload it
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            
        except Exception as e:
            raise CustomException(e, sys) 
    

    def get_df_from_object(self, object_: object) -> pd.DataFrame:
        """
        Converts an S3 object to a DataFrame.

        Args:
            object_ (object): The S3 object.

        Returns:
            DataFrame: DataFrame created from the object content.
        """ 
        try:
            content = self.read_object(object_, make_readable=True)
            df = pd.read_csv(content, na_values="na") 
            return df
        
        except Exception as e:
            raise CustomException(e, sys) 
        


    def read_csv(self, filename: str, bucket_name: str) -> pd.DataFrame:
        """
        Reads a CSV file from the specified S3 bucket and converts it to a DataFrame.

        Args:
            filename (str): The name of the file in the bucket.
            bucket_name (str): The name of the S3 bucket.

        Returns:
            DataFrame: DataFrame created from the CSV file.
        """ 
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj) 
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
