const fs = require('fs');
const path = require('path');

const categories = [
    {
        id: "ข.1",
        title: "ชุดคำสั่ง Backend Python Services",
        description: "ชุดคำสั่งนี้ทำหน้าที่เป็นจุดศูนย์กลาง (Core API) ของระบบหลังบ้าน โดยจัดการการรับส่งข้อมูลผ่าน RESTful API การเชื่อมต่อระหว่างระบบหน้าบ้านและเอนจินสแกนช่องโหว่ รวมถึงการตั้งค่าการทำงานต่างๆ ของเซิร์ฟเวอร์",
        files: [
            { path: 'backend/app/main.py', title: 'ชุดคำสั่งจัดการเส้นทาง (Router) และเริ่มต้นระบบ' }
        ]
    },
    {
        id: "ข.2",
        title: "ชุดคำสั่ง Frontend TypeScript React",
        description: "ชุดคำสั่งนี้รับผิดชอบการแสดงผลส่วนติดต่อผู้ใช้งาน (User Interface) การรับข้อมูลโค้ดจากผู้ใช้เพื่อส่งไปตรวจสอบ และการนำเสนอผลลัพธ์การสแกนในรูปแบบหน้าต่างที่เข้าใจง่าย",
        files: [
            { path: 'frontend/src/app/page.tsx', title: 'ชุดคำสั่งหน้าหลักของแอปพลิเคชัน' },
            { path: 'frontend/src/components/scanner/CodeInput.tsx', title: 'ชุดคำสั่งคอมโพเนนต์รับข้อมูลโค้ดเพื่อสแกน' },
            { path: 'frontend/src/components/scanner/ScanResults.tsx', title: 'ชุดคำสั่งคอมโพเนนต์แสดงผลการสแกนและรายงาน' }
        ]
    },
    {
        id: "ข.3",
        title: "ชุดคำสั่ง Hybrid Scanner และ AI Engine",
        description: "ชุดคำสั่งนี้เป็นหัวใจสำคัญของระบบ โดยรวบรวมตรรกะการทำงานของการสแกนช่องโหว่แบบผสมผสาน ทั้งการใช้ระบบกฎ (Pattern Matching) และปัญญาประดิษฐ์ (AI Model) ในการประมวลผล",
        files: [
            { path: 'backend/app/hybrid_scanner/pipeline.py', title: 'ชุดคำสั่งจัดการท่อส่งข้อมูลการสแกนหลัก' },
            { path: 'backend/app/hybrid_scanner/pattern_engine.py', title: 'ชุดคำสั่งเอนจินตรวจสอบด้วยกฎ' },
            { path: 'backend/app/ml/hybrid_model.py', title: 'ชุดคำสั่งโมเดลปัญญาประดิษฐ์' }
        ]
    },
    {
        id: "ข.4",
        title: "ชุดคำสั่ง Docker and Deployment",
        description: "ชุดคำสั่งนี้ใช้สำหรับการเตรียมและการจัดการสภาพแวดล้อมการทำงานของระบบ (Containerization) เพื่อให้สามารถนำโปรแกรมไปติดตั้งและทำงานบนเซิร์ฟเวอร์ได้อย่างสมบูรณ์",
        files: [
            { path: 'docker-compose.yml', title: 'ชุดคำสั่งตั้งค่าคอนเทนเนอร์ภาพรวมของระบบ' },
            { path: 'backend/Dockerfile', title: 'ชุดคำสั่งสร้างสภาพแวดล้อมสำหรับ Backend' }
        ]
    }
];

const basePath = 'c:\\Users\\user\\Desktop\\Project Final University Bon\\sp-project-academy-va-scan';
const outputPath = path.join(basePath, 'docs', 'APPENDIX_B_SOURCE_CODE.md');

let content = '# ภาคผนวก ข\n\n**ชุดคำสั่ง (Source Code)**\n\n\tภาคผนวกนี้รวบรวมชุดคำสั่ง (Source Code) ที่เป็นส่วนสำคัญของระบบตรวจสอบช่องโหว่ความปลอดภัยของเว็บแอปพลิเคชัน (Hybrid Vulnerability Scanner) โดยแบ่งออกตามโครงสร้างสถาปัตยกรรมของระบบ เพื่อใช้อ้างอิงการทำงาน โดยมีหัวข้อทั้งหมด ดังนี้\n';

for (const category of categories) {
    content += `\t${category.id} ${category.title}\n`;
}

content += '\n<br>\n\n';

for (const category of categories) {
    content += `**${category.id} ${category.title}**\n\n`;
    content += `${category.description}\n\n`;
    
    let fileIndex = 1;
    for (const fileObj of category.files) {
        const fullPath = path.join(basePath, fileObj.path.replace(/\//g, '\\'));
        if (fs.existsSync(fullPath)) {
            const fileContent = fs.readFileSync(fullPath, 'utf-8');
            const ext = fileObj.path.split('.').pop();
            let lang = ext;
            if (ext === 'py') lang = 'python';
            else if (ext === 'tsx' || ext === 'ts') lang = 'typescript';
            else if (ext === 'js') lang = 'javascript';
            else if (ext === 'yml') lang = 'yaml';
            content += `**${category.id}.${fileIndex} ${fileObj.title}**\n\n**ไฟล์:** \`${fileObj.path}\`\n\n\`\`\`${lang}\n${fileContent}\n\`\`\`\n\n`;
        } else {
            console.log('File not found: ' + fileObj.path);
        }
        fileIndex++;
    }
}

fs.writeFileSync(outputPath, content, 'utf-8');
console.log('Success! Created APPENDIX_B_SOURCE_CODE.md with formatted categories.');
