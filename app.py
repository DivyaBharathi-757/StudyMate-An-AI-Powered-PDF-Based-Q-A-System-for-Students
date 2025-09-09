def analyze_with_ibm_watson(text):
    """Analyzes text with IBM Watson NLU for named entity recognition."""
    if nlu is None:
        print("❌ IBM Watson NLU client not initialized. Skipping analysis.")
        return []
    try:
        response = nlu.analyze(
            text=text,
            features=Features(entities=EntitiesOptions(sentiment=False, emotion=False, limit=10))
        ).get_result()

        entities = response.get("entities", [])
        if not entities:
            print("\n💡 No entities found by IBM Watson.")
            return []
        else:
            print("\n💡 IBM Watson Entities:")
            for entity in entities:
                # Ensure 'type' and 'text' keys exist
                entity_type = entity.get('type', 'UNKNOWN')
                entity_text = entity.get('text', '')
                relevance = entity.get('relevance', 0.0)
                print(f"{entity_type}: {entity_text} (Relevance: {relevance:.2f})")
        return entities
    except Exception as e:
        print(f"❌ IBM Watson analysis error: {e}")
        return []


def verify_prescription(entities_hf, entities_ibm):
    """Verifies if the prescription contains drug name and dosage."""
    print("\n✅ Prescription Verification Results:")

    # Extract drug names from Hugging Face results
    # Assuming 'DRUG' and 'CHEMICAL' are the relevant entity groups for drugs
    drug_names = [
        ent['word'].lower()
        for ent in entities_hf
        if ent.get('entity_group', '').upper() in ['DRUG', 'CHEMICAL'] and 'word' in ent
    ]

    # Check for dosage patterns (e.g., "10 mg", "5 ml", "250mcg", "1g")
    # Expanded regex to include common units and be case-insensitive
    dosage_pattern = re.compile(r'\b\d+(\.\d+)?\s*(mg|ml|mcg|g|unit|units|drops|tablets|capsules)\b', re.IGNORECASE)
    dosage_found = any(
        dosage_pattern.search(ent['word'].lower())
        for ent in entities_hf if 'word' in ent
    )

    # Output results
    if drug_names:
        # Use set to show unique drug names
        print(f"✅ Drug(s) detected: {', '.join(sorted(list(set(drug_names))))}")
    else:
        print("⚠️ No drug names detected by Hugging Face.")

    if dosage_found:
        print("✅ Dosage information detected.")
    else:
        print("⚠️ Dosage information missing or not clearly identified.")

    # IBM Watson verification
    if not entities_ibm:
        print("💡 IBM Watson found no relevant medical entities (or was not initialized).")
    else:
        # You might want to filter IBM Watson entities for specific types related to medicine
        medical_entities_ibm = [
            ent for ent in entities_ibm
            if ent.get('type', '').lower() in ['medicine', 'drug', 'medical_condition', 'treatment'] # Example types
        ]
        if medical_entities_ibm:
            print(f"💡 IBM Watson identified {len(medical_entities_ibm)} potential medical entity(ies):")
            for ent in medical_entities_ibm:
                print(f"  - {ent.get('type')}: {ent.get('text')} (Relevance: {ent.get('relevance'):.2f})")
        else:
            print("💡 IBM Watson did not identify specific medical entities among the top results.")


def main():
    # === Replace this path with your prescription image ===
    image_path = "prescription.jpeg" # Make sure this file exists!

    # Step 1: Extract text from image
    extracted_text = extract_text(image_path)
    if not extracted_text.strip():
        print("❌ No text extracted from the image. Please check the file path and image quality.")
        return

    # Step 2: Analyze using Hugging Face
    hf_entities = analyze_with_huggingface(extracted_text)

    # Step 3: Analyze using IBM Watson
    ibm_entities = analyze_with_ibm_watson(extracted_text)

    # Step 4: Verify presence of medical info
    verify_prescription(hf_entities, ibm_entities)


if __name__ == "__main__":
    main()

