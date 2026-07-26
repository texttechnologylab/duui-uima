from pydantic import BaseModel, Field
from typing import Literal

type TaxonProvider = Literal["gbif", "taxref", "ncbi"]

class SharedTaxon(BaseModel):
    provider: TaxonProvider
    taxon_id: int
    kingdom_name: str | None = Field(default=None)
    kingdom_key: int | None = Field(default=None)
    phylum_name: str | None = Field(default=None)
    phylum_key: int | None = Field(default=None)
    class_name: str | None = Field(default=None)
    class_key: int | None = Field(default=None)
    order_name: str | None = Field(default=None)
    order_key: int | None = Field(default=None)
    superfamily_name: str | None = Field(default=None)
    superfamily_key: int | None = Field(default=None)
    family_name: str | None = Field(default=None)
    family_key: int | None = Field(default=None)
    subfamily_name: str | None = Field(default=None)
    subfamily_key: int | None = Field(default=None)
    tribe_name: str | None = Field(default=None)
    tribe_key: int | None = Field(default=None)
    subtribe_name: str | None = Field(default=None)
    subtribe_key: int | None = Field(default=None)
    genus_name: str | None = Field(default=None)
    genus_key: int | None = Field(default=None)
    subgenus_name: str | None = Field(default=None)
    subgenus_key: int | None = Field(default=None)
    species_name: str | None = Field(default=None)
    species_key: int | None = Field(default=None)
    parent_name: str | None = Field(default=None)
    parent_key: int | None = Field(default=None)
    scientific_name: str | None = Field(default=None)
    canonical_name: str | None = Field(default=None)
    vernacular_name: str | None = Field(default=None)
    accepted_name_usage: str | None = Field(default=None)
    authorship: str | None = Field(default=None)
    name_type: str | None = Field(default=None)
    rank: str | None = Field(default=None)
    origin: str | None = Field(default=None)
    taxonomic_status: str | None = Field(default=None)
    remarks: str | None = Field(default=None)
    references: str | None = Field(default=None)
    published_in: str | None = Field(default=None)
    num_descendants: int | None = Field(default=None)
    last_crawled: str | None = Field(default=None)
    last_interpreted: str | None = Field(default=None)
    species_epithet: str | None = Field(default=None)
    infraspecific_epithet: str | None = Field(default=None)
    cultivar_epithet: str | None = Field(default=None)
    url: str | None = Field(default=None)
    wikidata_id: str | None = Field(default=None)
    wikidata_url: str | None = Field(default=None)

class TaxonBase:
    def as_shared(self) -> SharedTaxon:
        raise NotImplementedError("Subclasses must implement as_shared method")

