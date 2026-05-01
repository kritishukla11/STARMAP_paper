import requests

TOKEN = "OL2ARi3pWoQJ73zW1PJUo2oDS9voRtFejlAol8PWlg8kHoOZSJb6t8NdtqABmACFFE8zPoay7FyuEynuRkg1PolvE6HjIOq8BgWHZKyC2av6cAgf4se/WwOBuQo"
collection = "TFT:GTRD"
version = "v2024.1.Hs"

url = f"https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/filesets/{version}/c3.tft.gtrd.{version}.symbols.gmt"

headers = {
    "User-Agent": "Python script",
    "Cookie": f"MSIGDBUserToken={TOKEN}"
}

response = requests.get(url, headers=headers)

with open(f"c3.tft.gtrd.{version}.symbols.gmt", "wb") as f:
    f.write(response.content)

print("Downloaded TFT:GTRD gene set!")