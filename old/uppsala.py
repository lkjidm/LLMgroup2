import json
import os
import xml.etree.ElementTree as ET

osm_file_path = "knowledge/map"  # 如果改名了可填写 "map.xml" 或 "map.osm"
output_json_path = "knowledge/uppsala_all_places.json"

os.makedirs("knowledge", exist_ok=True)

print(f"正在解析 OSM 文件并提取所有命名节点: {osm_file_path} ...")

try:
    tree = ET.parse(osm_file_path)
    root = tree.getroot()

    locations = []
    seen_names = set()

    for node in root.findall("node"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}
        
        # 只要节点带有 name 标签，就提取出来
        name = tags.get("name")
        if name and name not in seen_names:
            seen_names.add(name)
            
            # 自动识别节点的分类类型（优先级依次递减）
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

    # 写入 JSON 知识库文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(locations, f, ensure_ascii=False, indent=2)

    print(f"🎉 成功！共提取出 {len(locations)} 个完整地名/地标节点，已保存至：{output_json_path}")

except Exception as e:
    print(f"❌ 解析失败，错误信息: {e}")