import json
import os
import xml.etree.ElementTree as ET

osm_file_path = "knowledge/map"  # Use "map.xml" or "map.osm" here if the file was renamed
output_json_path = "knowledge/uppsala_all_places.json"

os.makedirs("knowledge", exist_ok=True)

print(f"Parsing the OSM file and extracting named nodes: {osm_file_path} ...")

try:
    tree = ET.parse(osm_file_path)
    root = tree.getroot()

    locations = []
    seen_names = set()

    for node in root.findall("node"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}
        
        # Extract nodes that have a name tag
        name = tags.get("name")
        if name and name not in seen_names:
            seen_names.add(name)
            
            # Identify the place type using the following priority order
            place_type = (
                tags.get("place") or 
                tags.get("amenity") or 
                tags.get("building") or 
                tags.get("leisure") or 
                tags.get("landuse") or 
                tags.get("railway") or 
                "landmark"
            )
            
            lat = float(node.attrib["lat"])
            lon = float(node.attrib["lon"])

            locations.append({
                "name": name,
                "type": place_type,
                "city": "Uppsala",
                "lat": lat,
                "lon": lon,
                "description": f"{name} is a local {place_type} in Uppsala."
            })

    # Write the JSON knowledge file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(locations, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(locations)} named place/landmark nodes and saved them to {output_json_path}.")

except Exception as e:
    print(f"Parsing failed: {e}")