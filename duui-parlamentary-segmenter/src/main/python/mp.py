import pandas as pd
from nameparser import HumanName

def get_mp(legislative_period:int):
    # URLs der Wahlperioden 1–8
    url = (
        "https://de.wikipedia.org/wiki/"
        f"Liste_der_Reichstagsabgeordneten_der_Weimarer_Republik_({legislative_period}._Wahlperiode)"
    )

    tables = pd.read_html(url)

    # Wähle die relevante Tabelle
    correct = next(
        (tbl for tbl in tables if "Abgeordneter" in tbl.columns and len(tbl) >= 300),
        tables[0]
    )
    df = correct.copy()
    df["Legislaturperiode"] = legislative_period

    # Funktion zum Aufsplitten des Namens
    def split_name(full):
        hn = HumanName(full)

        if hn.middle in ['Graf', 'Herzog', 'Freiherr']:
            nobility = hn.middle
            middle_name = None
        else:
            middle_name = hn.middle
            nobility = None

        return pd.Series({
            "title": hn.title or None,
            "nobility": nobility or None,
            "first_name": hn.first or None,
            "middle_name": middle_name or None,
            "family_name": hn.last or None,
        })

    # Anwenden auf die Spalte "Abgeordneter"
    name_parts = df["Abgeordneter"].astype(str).apply(split_name)
    df = pd.concat([df, name_parts], axis=1)

    return df
