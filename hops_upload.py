import hopsworks

project = hopsworks.login()

dataset_api = project.get_dataset_api()

dataset_api.upload("./demo_sentiment_history.png", "Resources/images", overwrite=True)
dataset_api.upload("./general_sentiment_history.png", "Resources/images", overwrite=True)
dataset_api.upload("./news_image.png", "Resources/images", overwrite=True)
