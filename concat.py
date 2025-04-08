import pandas as pd
from typing import Optional

def df_append(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if old is None:
        old = new

    else:
        old = pd.concat([old, new], axis=0)

    return old