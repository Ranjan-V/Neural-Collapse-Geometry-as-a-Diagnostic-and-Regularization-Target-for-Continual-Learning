Real split manifests require the attached Tiny ImageNet class IDs. Before any
scientific job, Kaggle binds all three deterministic recipes to that dataset
and writes outputs/splits/seed_42.json, seed_43.json and seed_44.json. These
contain actual class IDs, WordNet IDs, class order and task memberships.
Every resume checks their hash and refuses changed class mappings.
