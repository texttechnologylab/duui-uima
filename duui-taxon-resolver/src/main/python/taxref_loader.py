import os
import tempfile
from typing import override
from pydantic import BaseModel, Field
import requests
import zipfile

import pandas as pd

### SETUP ###


def download_backbone(
    output_path: str, url: str = "https://ipt.gbif.fr/archive.do?r=taxref"
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        # download the zip file
        response = requests.get(url)
        response.raise_for_status()
        # write the content to the temporary file
        tmp.write(response.content)
        tmp.flush()
        # extract the zip file
        with zipfile.ZipFile(tmp.name, "r") as zip_ref:
            zip_ref.extractall(output_path)


def load_backbone(dir_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    vernacular_names_path = f"{dir_path}/vernacularname.txt"
    taxonomy_path = f"{dir_path}/taxon.txt"
    vernacular_names = pd.read_csv(vernacular_names_path, sep="\t", low_memory=False)
    taxonomy = pd.read_csv(taxonomy_path, sep="\t", low_memory=False)
    return vernacular_names, taxonomy


def load_backbone_from_url(
    url: str = "https://ipt.gbif.fr/archive.do?r=taxref&v=4.17",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with tempfile.TemporaryDirectory() as tmpdir:
        download_backbone(tmpdir, url)
        return load_backbone(tmpdir)


def load_taxref() -> tuple[pd.DataFrame, pd.DataFrame]:
    local_path = "backbone"
    if not os.path.exists(local_path):
        return load_backbone_from_url()
    else:
        return load_backbone(local_path)


### Backbone data ###

vernacular_names, taxonomy = load_taxref()

### Utility methods ###


def taxon_index(taxon_id: int) -> int:
    return taxonomy.index[taxonomy["taxonID"] == taxon_id][0]


def vernacular_name_index(vernacular_name_id: int) -> int:
    return vernacular_names.index[vernacular_names["id"] == vernacular_name_id][0]


### Wrapper classes ###


class TaxrefTaxon(BaseModel):
    id_: int = Field(alias="id")
    taxon_id: int = Field(alias="taxonID")
    scientific_name_id: int | None = Field(alias="scientificNameID", default=None)
    accepted_name_usage_id: int | None = Field(
        alias="acceptedNameUsageID", default=None
    )
    parent_name_usage_id: int | None = Field(alias="parentNameUsageID", default=None)
    original_name_usage_id: int | None = Field(
        alias="originalNameUsageID", default=None
    )
    scientific_name: str | None = Field(alias="scientificName", default=None)
    accepted_name_usage: str | None = Field(alias="acceptedNameUsage", default=None)
    kingdom: str | None = Field(default=None)
    phylum: str | None = Field(default=None)
    class_: str | None = Field(alias="class", default=None)
    order: str | None = Field(default=None)
    superfamily: str | None = Field(default=None)
    family: str | None = Field(default=None)
    subfamily: str | None = Field(default=None)
    tribe: str | None = Field(default=None)
    subtribe: str | None = Field(default=None)
    genus: str | None = Field(default=None)
    subgenus: str | None = Field(default=None)
    specific_epithet: str | None = Field(alias="specificEpithet", default=None)
    infraspecific_epithet: str | None = Field(
        alias="infraspecificEpithet", default=None
    )
    cultivar_epithet: str | None = Field(alias="cultivarEpithet", default=None)
    taxon_rank: str | None = Field(alias="taxonRank", default=None)
    scientific_name_authorship: str | None = Field(
        alias="scientificNameAuthorship", default=None
    )
    vernacular_name: str | None = Field(alias="vernacularName", default=None)
    taxon_remarks: str | None = Field(alias="taxonRemarks", default=None)
    references: str | None = Field(default=None)


class VernacularName(BaseModel):
    id_: int = Field(alias="id")
    vernacular_name: str = Field(alias="vernacularName")
    source: str | None = Field(default=None)
    language: str | None = Field(default=None)
    location_id: str | None = Field(alias="locationID", default=None)
    country_code: str | None = Field(alias="countryCode", default=None)


def taxon_from_id(taxon_id: int) -> TaxrefTaxon:
    taxon_index_ = taxon_index(taxon_id)
    taxon_data = taxonomy.loc[taxon_index_]
    # convert NaN to None for optional fields
    taxon_data = taxon_data.where(pd.notnull(taxon_data), None)
    return TaxrefTaxon(**taxon_data)


def vernacular_name_from_id(vernacular_name_id: int) -> VernacularName:
    vernacular_name_index_ = vernacular_name_index(vernacular_name_id)
    vernacular_name_data = vernacular_names.loc[vernacular_name_index_]
    # convert NaN to None for optional fields
    vernacular_name_data = vernacular_name_data.where(
        pd.notnull(vernacular_name_data), None
    )
    return VernacularName(**vernacular_name_data)


def main():
    while True:
        print(
            "Enter taxon ID 't {id}' or vernacular name ID 'v {id}' (or 'exit' to quit): ",
            end="",
        )
        user_input = input().strip()
        if user_input.lower() == "exit":
            break
        if user_input.startswith("t "):
            taxon_id_str = user_input[2:].strip()
            try:
                taxon_id = int(taxon_id_str)
                taxon = taxon_from_id(taxon_id)
                print(taxon)
            except ValueError:
                print(
                    f"Invalid taxon ID '{taxon_id_str}'. Please enter a valid integer taxon ID."
                )
            except IndexError:
                print("Taxon ID not found. Please enter a valid taxon ID.")
        elif user_input.startswith("v "):
            vernacular_name_id_str = user_input[2:].strip()
            try:
                vernacular_name_id = int(vernacular_name_id_str)
                vernacular_name = vernacular_name_from_id(vernacular_name_id)
                print(vernacular_name)
            except ValueError:
                print(
                    f"Invalid vernacular name ID '{vernacular_name_id_str}'. Please enter a valid integer vernacular name ID."
                )
            except IndexError:
                print(
                    "Vernacular name ID not found. Please enter a valid vernacular name ID."
                )


if __name__ == "__main__":
    main()
