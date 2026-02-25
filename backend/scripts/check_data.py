import torch
import pickle
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# --- ส่วนสำคัญ: ต้องนิยาม Class นี้ให้เหมือนกับตอนสร้างข้อมูล ---
@dataclass
class ProcessedSample:
    """Processed code sample with graph representation"""
    code: str
    label: int  # 0 = safe, 1 = vulnerable
    language: str
    graph_data: any  # PyTorch Geometric Data object with x, edge_index, edge_attr, y
    vulnerability_type: str
    source: str
    metadata: Dict
    token_ids: Optional[torch.Tensor] = None  # Token sequence for LSTM [1, seq_len]
    code_metrics: Optional[np.ndarray] = None  # 20 code metrics features

# --- ส่วนตรวจสอบข้อมูล ---
def check_data():
    # ระบุไฟล์ที่ต้องการเช็ก (Relative Path จากโฟลเดอร์ backend)
    file_path = "data/processed_graphs/train_graphs.pkl" 
    
    print(f"📂 กำลังตรวจสอบไฟล์ที่: {file_path}")
    print(f"📍 ตำแหน่งปัจจุบันของคุณ: {os.getcwd()}")
    
    if not os.path.exists(file_path):
        print(f"❌ ไม่เจอไฟล์! ลองเช็กว่าในโฟลเดอร์ backend/data/processed_graphs มีไฟล์ชื่อ train_graphs.pkl ไหม")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"✅ อ่านไฟล์สำเร็จ! เจอข้อมูลทั้งหมด {len(data)} ตัวอย่าง")
        
        # ดึงตัวอย่างแรกมาเช็ก
        sample = data[0]
        print("\n--- 🔍 ผลการตรวจ Sample 0 ---")
        print(f"📝 Language: {sample.language}")
        print(f"🏷️  Label: {'Vulnerable' if sample.label == 1 else 'Safe'}")
        print(f"🔧 Vulnerability Type: {sample.vulnerability_type}")
        print(f"📦 Source: {sample.source}")
        
        try:
            # เข้าถึง graph features ผ่าน graph_data
            x = sample.graph_data.x
            print(f"\n📊 Feature Shape: {x.shape}")
            print(f"📈 Max Value: {x.max().item():.6f}")
            print(f"📉 Min Value: {x.min().item():.6f}")
            
            # เช็กว่าเป็น 0 ทั้งหมดหรือไม่
            is_all_zeros = (x == 0).all().item()
            print(f"🔍 Is All Zeros: {is_all_zeros}")
            
            if is_all_zeros:
                print("\n😱 ⚠️ ข้อมูลเป็น 0 ทั้งหมด! (CodeBERT embeddings ไม่ทำงาน)")
                print("👉 วิธีแก้: ต้องแก้ไขการสร้าง embeddings ใน pipeline")
            else:
                print("\n🎉 ✅ ข้อมูลปกติครับ! (มีค่า features)")
                
            # แสดงข้อมูล graph structure
            print(f"\n📐 Graph Structure:")
            print(f"   - Nodes: {sample.graph_data.x.shape[0]}")
            print(f"   - Edges: {sample.graph_data.edge_index.shape[1]}")
            if hasattr(sample.graph_data, 'edge_attr') and sample.graph_data.edge_attr is not None:
                print(f"   - Edge Features: {sample.graph_data.edge_attr.shape}")
            if hasattr(sample.graph_data, 'y') and sample.graph_data.y is not None:
                print(f"   - Label (y): {sample.graph_data.y.item()}")
                
        except AttributeError as e:
            print(f"\n⚠️ ไม่สามารถเข้าถึง graph_data: {e}")
            print("Structure อาจไม่ตรงกับที่คาดหวัง")
        except Exception as e:
            print(f"\n⚠️ Error ตอนวิเคราะห์ข้อมูล: {e}")

    except Exception as e:
        print(f"\n❌ Error ตอนโหลด: {e}")

if __name__ == "__main__":
    check_data()