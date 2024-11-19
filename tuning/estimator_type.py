from dto.dto import RequestDTO
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_estimator(request: RequestDTO):
    switcher = {
        "classification": RandomForestClassifier(),
        "regression": RandomForestRegressor()
    }
    return switcher.get(request.task.value, None)
