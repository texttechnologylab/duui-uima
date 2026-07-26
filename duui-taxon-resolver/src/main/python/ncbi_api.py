import taxoniq
from typing import List, Self, override
from shared_taxon import SharedTaxon, TaxonBase

class NcbiTaxon(TaxonBase):
    handle: taxoniq.Taxon

    def __init__(self, handle: taxoniq.Taxon):
        self.handle = handle

    @classmethod
    def from_tax_id(cls, tax_id: int) -> Self:
        handle = taxoniq.Taxon(tax_id)
        return cls(handle)

    @property
    def taxon_id(self) -> int | None:
        return self.handle.tax_id
    
    @property
    def rank(self) -> str | None:
        try:
            enum_rank = self.handle.rank
            return enum_rank.name if enum_rank is not None else None
        except taxoniq.NoValue:
            return None
    
    @property
    def scientific_name(self) -> str | None:
        try:
            return self.handle.scientific_name
        except taxoniq.NoValue:
            return None
    
    @property
    def common_name(self) -> str | None:
        try:
            return self.handle.common_name
        except taxoniq.NoValue:
            return None
    
    @property
    def lineage(self) -> List[Self] | None:
        try:
            return [NcbiTaxon(taxon) for taxon in self.handle.lineage]
        except taxoniq.NoValue:
            return None
        
    @property
    def ranked_lineage(self) -> List[Self] | None:
        try:
            return [NcbiTaxon(taxon) for taxon in self.handle.ranked_lineage]
        except taxoniq.NoValue:
            return None
    
    @property
    def parent(self) -> Self | None:
        try:
            return NcbiTaxon(self.handle.parent)
        except taxoniq.NoValue:
            return None
    
    @property
    def description(self) -> str | None:
        try:
            return self.handle.description
        except taxoniq.NoValue:
            return None
    
    @property
    def url(self) -> str:
        return self.handle.url
    
    @property
    def wikidata_id(self) -> str | None:
        try:
            return self.handle.wikidata_id
        except KeyError:
            return None

    @property
    def wikidata_url(self) -> str | None:
        try:
            return self.handle.wikidata_url
        except KeyError:
            return None

    @override
    def as_shared(self) -> SharedTaxon:
        tid = self.taxon_id
        if tid is None:
            raise ValueError("Taxon ID is required to convert to SharedTaxon")
        parent = self.parent
        parent_name = parent.scientific_name if parent is not None else None
        parent_key = parent.taxon_id if parent is not None else None
        return SharedTaxon(
            provider="ncbi",
            taxon_id=tid,
            scientific_name=self.scientific_name,
            vernacular_name=self.common_name,
            parent_name=parent_name,
            parent_key=parent_key,
            rank=self.rank,
            remarks=self.description,
            url=self.url,
            wikidata_id=self.wikidata_id,
            wikidata_url=self.wikidata_url,
        )
