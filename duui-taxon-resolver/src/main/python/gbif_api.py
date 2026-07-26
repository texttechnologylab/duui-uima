from typing import override

from pydantic import BaseModel, Field
import requests
from shared_taxon import SharedTaxon, TaxonBase

base_api_url = "https://api.gbif.org/v1"


class GbifTaxon(BaseModel, TaxonBase):
    key: int
    taxon_id: str = Field(alias="taxonID")
    kingdom: str | None = Field(default=None)
    order: str | None = Field(default=None)
    family: str | None = Field(default=None)
    genus: str | None = Field(default=None)
    species: str | None = Field(default=None)
    kingdom_key: int | None = Field(alias="kingdomKey", default=None)
    order_key: int | None = Field(alias="orderKey", default=None)
    family_key: int | None = Field(alias="familyKey", default=None)
    genus_key: int | None = Field(alias="genusKey", default=None)
    species_key: int | None = Field(alias="speciesKey", default=None)
    parent_key: int | None = Field(alias="parentKey", default=None)
    parent: str | None = Field(default=None)
    scientific_name: str | None = Field(alias="scientificName", default=None)
    canonical_name: str | None = Field(alias="canonicalName", default=None)
    vernacular_name: str | None = Field(alias="vernacularName", default=None)
    authorship: str | None = Field(default=None)
    name_type: str | None = Field(alias="nameType", default=None)
    rank: str
    origin: str | None = Field(default=None)
    taxonomic_status: str | None = Field(alias="taxonomicStatus", default=None)
    remarks: str | None = Field(default=None)
    published_in: str | None = Field(alias="publishedIn", default=None)
    num_descendants: int | None = Field(alias="numDescendants", default=None)
    last_crawled: str | None = Field(alias="lastCrawled", default=None)
    last_interpreted: str | None = Field(alias="lastInterpreted", default=None)
    issues: list[str] = Field(default_factory=list)
    class_: str | None = Field(alias="class", default=None)

    @property
    def raw_taxon_id(self) -> int:
        return int(self.taxon_id.split(":")[-1])

    @override
    def as_shared(self) -> SharedTaxon:
        return SharedTaxon(
            provider="gbif",
            taxon_id=self.raw_taxon_id,
            kingdom_name=self.kingdom,
            kingdom_key=self.kingdom_key,
            order_name=self.order,
            order_key=self.order_key,
            family_name=self.family,
            family_key=self.family_key,
            genus_name=self.genus,
            genus_key=self.genus_key,
            species_name=self.species,
            species_key=self.species_key,
            parent_name=self.parent,
            parent_key=self.parent_key,
            scientific_name=self.scientific_name,
            canonical_name=self.canonical_name,
            vernacular_name=self.vernacular_name,
            authorship=self.authorship,
            name_type=self.name_type,
            rank=self.rank,
            origin=self.origin,
            taxonomic_status=self.taxonomic_status,
            remarks=self.remarks,
            published_in=self.published_in,
            num_descendants=self.num_descendants,
            last_crawled=self.last_crawled,
            last_interpreted=self.last_interpreted,
            url=f"https://www.gbif.org/species/{self.key}",
        )


def get_taxon(taxon_id: int) -> GbifTaxon:
    response = requests.get(f"{base_api_url}/species/{taxon_id}")
    response.raise_for_status()
    response_data = response.content
    return GbifTaxon.model_validate_json(response_data)


def main():
    while True:
        print("Enter taxon ID (or 'exit' to quit): ", end="")
        user_input = input().strip()
        if user_input.lower() == "exit":
            break
        try:
            taxon_id = int(user_input)
        except ValueError:
            print(
                f"Invalid input '{user_input}'. Please enter a valid integer taxon ID."
            )
            continue
        try:
            taxon = get_taxon(taxon_id)
            print(taxon)
        except ValueError as e:
            print(f"Error parsing taxon data: {e}")
        except requests.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
