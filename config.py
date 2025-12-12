from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings class using Pydantic Settings.
    It automatically reads environment variables (e.g., from AWS Lambda), 
    and falls back to the .env file for local development.
    """

    # APP SETTINGS
    ENVIRONMENT: str = "production"
    UVICORN_PORT: int = 8000

    # AWS/SAGEMAKER CONFIGURATION
    SAGEMAKER_ENDPOINT_NAME: str
    AWS_REGION: str = "ap-south-1"

    # AWS Credentials for Boto3 (Lambda will ignore these and use IAM Role)
    # They are included here for local Boto3 testing convenience
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None

    # Configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore"
    )
# Instantiate the settings object globally
settings = Settings()

# Example usage check
print(f"Loaded Endpoint Name: {settings.SAGEMAKER_ENDPOINT_NAME}")