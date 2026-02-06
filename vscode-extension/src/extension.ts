import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';

// ==================== Types ====================

interface VulnerabilityFinding {
    line: number;
    column?: number;
    type: string;
    severity: string;
    message: string;
    cwe?: string;
    owasp?: string;
    suggestion?: string;
    confidence?: number;
}

interface ScanResult {
    findings: VulnerabilityFinding[];
    scan_time_ms: number;
    ml_enabled: boolean;
    pattern_version: string;
}

interface ScanRequest {
    code: string;
    language?: string;
    filename?: string;
    include_ml?: boolean;
}

// ==================== Diagnostic Collection ====================

let diagnosticCollection: vscode.DiagnosticCollection;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;
let apiClient: AxiosInstance;

// ==================== Extension Activation ====================

export function activate(context: vscode.ExtensionContext) {
    console.log('AI Vulnerability Scanner extension activated');

    // Initialize components
    diagnosticCollection = vscode.languages.createDiagnosticCollection('aiVulnScanner');
    outputChannel = vscode.window.createOutputChannel('AI Vulnerability Scanner');
    
    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'aiVulnScanner.scanCurrentFile';
    statusBarItem.text = '$(shield) Scan';
    statusBarItem.tooltip = 'Click to scan current file for vulnerabilities';
    statusBarItem.show();

    // Initialize API client
    initializeApiClient();

    // Register commands
    const commands = [
        vscode.commands.registerCommand('aiVulnScanner.scanCurrentFile', scanCurrentFile),
        vscode.commands.registerCommand('aiVulnScanner.scanWorkspace', scanWorkspace),
        vscode.commands.registerCommand('aiVulnScanner.scanSelection', scanSelection),
        vscode.commands.registerCommand('aiVulnScanner.showDashboard', showDashboard),
        vscode.commands.registerCommand('aiVulnScanner.configure', openSettings),
    ];

    commands.forEach(cmd => context.subscriptions.push(cmd));
    context.subscriptions.push(diagnosticCollection);
    context.subscriptions.push(statusBarItem);
    context.subscriptions.push(outputChannel);

    // Register auto-scan on save
    const config = vscode.workspace.getConfiguration('aiVulnScanner');
    if (config.get<boolean>('autoScan')) {
        context.subscriptions.push(
            vscode.workspace.onDidSaveTextDocument(document => {
                if (isSupportedLanguage(document.languageId)) {
                    setTimeout(() => scanDocument(document), config.get<number>('autoScanDelay', 1000));
                }
            })
        );
    }

    // Register CodeLens provider
    if (config.get<boolean>('enableCodeLens')) {
        context.subscriptions.push(
            vscode.languages.registerCodeLensProvider(
                getSupportedLanguages(),
                new VulnerabilityCodeLensProvider()
            )
        );
    }

    outputChannel.appendLine('AI Vulnerability Scanner initialized');
}

export function deactivate() {
    diagnosticCollection.dispose();
    statusBarItem.dispose();
    outputChannel.dispose();
}

// ==================== API Client ====================

function initializeApiClient() {
    const config = vscode.workspace.getConfiguration('aiVulnScanner');
    const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');
    const apiKey = config.get<string>('apiKey', '');

    apiClient = axios.create({
        baseURL: serverUrl,
        timeout: 30000,
        headers: apiKey ? { 'X-API-Key': apiKey } : {}
    });
}

// ==================== Scanning Functions ====================

async function scanCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    await scanDocument(editor.document);
}

async function scanDocument(document: vscode.TextDocument) {
    if (!isSupportedLanguage(document.languageId)) {
        vscode.window.showInformationMessage(`Language ${document.languageId} is not supported`);
        return;
    }

    statusBarItem.text = '$(loading~spin) Scanning...';
    
    try {
        const code = document.getText();
        const config = vscode.workspace.getConfiguration('aiVulnScanner');
        
        const request: ScanRequest = {
            code,
            language: mapLanguageId(document.languageId),
            filename: document.fileName,
            include_ml: config.get<boolean>('useMLModel', true)
        };

        outputChannel.appendLine(`Scanning: ${document.fileName}`);
        
        const response = await apiClient.post<ScanResult>('/api/v1/scan', request);
        const result = response.data;

        // Convert to diagnostics
        const diagnostics = convertToDiagnostics(result.findings, document);
        diagnosticCollection.set(document.uri, diagnostics);

        // Update status bar
        const vulnCount = result.findings.length;
        if (vulnCount > 0) {
            statusBarItem.text = `$(shield) ${vulnCount} issues`;
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        } else {
            statusBarItem.text = '$(shield) No issues';
            statusBarItem.backgroundColor = undefined;
        }

        outputChannel.appendLine(`Found ${vulnCount} vulnerabilities in ${result.scan_time_ms}ms`);
        
        if (vulnCount > 0) {
            vscode.window.showWarningMessage(
                `Found ${vulnCount} potential vulnerabilities`,
                'Show Problems'
            ).then(action => {
                if (action === 'Show Problems') {
                    vscode.commands.executeCommand('workbench.action.problems.focus');
                }
            });
        }

    } catch (error: any) {
        outputChannel.appendLine(`Scan error: ${error.message}`);
        statusBarItem.text = '$(shield) Error';
        
        if (error.code === 'ECONNREFUSED') {
            vscode.window.showErrorMessage('Cannot connect to scanner server. Is it running?');
        } else {
            vscode.window.showErrorMessage(`Scan failed: ${error.message}`);
        }
    }
}

async function scanSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }

    const selection = editor.selection;
    if (selection.isEmpty) {
        vscode.window.showWarningMessage('No text selected');
        return;
    }

    const selectedText = editor.document.getText(selection);
    
    try {
        const config = vscode.workspace.getConfiguration('aiVulnScanner');
        
        const response = await apiClient.post<ScanResult>('/api/v1/scan', {
            code: selectedText,
            language: mapLanguageId(editor.document.languageId),
            include_ml: config.get<boolean>('useMLModel', true)
        });

        const vulnCount = response.data.findings.length;
        
        if (vulnCount > 0) {
            vscode.window.showWarningMessage(`Found ${vulnCount} vulnerabilities in selection`);
        } else {
            vscode.window.showInformationMessage('No vulnerabilities found in selection');
        }
        
    } catch (error: any) {
        vscode.window.showErrorMessage(`Scan failed: ${error.message}`);
    }
}

async function scanWorkspace() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Scanning workspace for vulnerabilities',
        cancellable: true
    }, async (progress, token) => {
        const files = await vscode.workspace.findFiles(
            '**/*.{py,js,ts,java,php,go,rb,cs}',
            '**/node_modules/**'
        );

        let scanned = 0;
        let totalVulns = 0;

        for (const file of files) {
            if (token.isCancellationRequested) {
                break;
            }

            progress.report({
                message: `${scanned}/${files.length} files`,
                increment: (1 / files.length) * 100
            });

            try {
                const document = await vscode.workspace.openTextDocument(file);
                const code = document.getText();
                
                const response = await apiClient.post<ScanResult>('/api/v1/scan', {
                    code,
                    language: mapLanguageId(document.languageId),
                    filename: file.fsPath
                });

                const diagnostics = convertToDiagnostics(response.data.findings, document);
                diagnosticCollection.set(file, diagnostics);
                totalVulns += response.data.findings.length;
                
            } catch (error) {
                outputChannel.appendLine(`Failed to scan ${file.fsPath}`);
            }

            scanned++;
        }

        vscode.window.showInformationMessage(
            `Scanned ${scanned} files, found ${totalVulns} vulnerabilities`
        );
    });
}

// ==================== Dashboard ====================

function showDashboard() {
    const panel = vscode.window.createWebviewPanel(
        'aiVulnScannerDashboard',
        'Vulnerability Dashboard',
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    panel.webview.html = getDashboardHtml();
}

function getDashboardHtml(): string {
    return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vulnerability Dashboard</title>
    <style>
        body { font-family: var(--vscode-font-family); padding: 20px; }
        .header { font-size: 24px; margin-bottom: 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 30px; }
        .stat-card { 
            padding: 20px; 
            background: var(--vscode-editor-background); 
            border: 1px solid var(--vscode-panel-border);
            border-radius: 8px;
            min-width: 150px;
        }
        .stat-value { font-size: 32px; font-weight: bold; }
        .stat-label { color: var(--vscode-descriptionForeground); }
        .severity-critical { color: #ff6b6b; }
        .severity-high { color: #ffa502; }
        .severity-medium { color: #ffcc00; }
        .severity-low { color: #26de81; }
    </style>
</head>
<body>
    <div class="header">🛡️ AI Vulnerability Scanner Dashboard</div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value" id="totalScans">0</div>
            <div class="stat-label">Total Scans</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="totalVulns">0</div>
            <div class="stat-label">Vulnerabilities Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="filesScanned">0</div>
            <div class="stat-label">Files Scanned</div>
        </div>
    </div>
    
    <h3>Severity Breakdown</h3>
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value severity-critical" id="critical">0</div>
            <div class="stat-label">Critical</div>
        </div>
        <div class="stat-card">
            <div class="stat-value severity-high" id="high">0</div>
            <div class="stat-label">High</div>
        </div>
        <div class="stat-card">
            <div class="stat-value severity-medium" id="medium">0</div>
            <div class="stat-label">Medium</div>
        </div>
        <div class="stat-card">
            <div class="stat-value severity-low" id="low">0</div>
            <div class="stat-label">Low</div>
        </div>
    </div>
</body>
</html>`;
}

// ==================== CodeLens Provider ====================

class VulnerabilityCodeLensProvider implements vscode.CodeLensProvider {
    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
        const codeLenses: vscode.CodeLens[] = [];
        const diagnostics = diagnosticCollection.get(document.uri) || [];

        for (const diagnostic of diagnostics) {
            const range = new vscode.Range(diagnostic.range.start.line, 0, diagnostic.range.start.line, 0);
            const command: vscode.Command = {
                title: `⚠️ ${diagnostic.message}`,
                command: 'editor.action.showHover',
                arguments: [{ position: diagnostic.range.start }]
            };
            codeLenses.push(new vscode.CodeLens(range, command));
        }

        return codeLenses;
    }
}

// ==================== Helper Functions ====================

function convertToDiagnostics(findings: VulnerabilityFinding[], document: vscode.TextDocument): vscode.Diagnostic[] {
    return findings.map(finding => {
        const line = Math.max(0, finding.line - 1);
        const lineText = document.lineAt(line).text;
        const range = new vscode.Range(line, 0, line, lineText.length);

        const severity = mapSeverity(finding.severity);
        
        const diagnostic = new vscode.Diagnostic(range, finding.message, severity);
        diagnostic.source = 'AI Vuln Scanner';
        diagnostic.code = finding.cwe || finding.type;
        
        return diagnostic;
    });
}

function mapSeverity(severity: string): vscode.DiagnosticSeverity {
    switch (severity.toLowerCase()) {
        case 'critical':
        case 'high':
            return vscode.DiagnosticSeverity.Error;
        case 'medium':
            return vscode.DiagnosticSeverity.Warning;
        case 'low':
            return vscode.DiagnosticSeverity.Information;
        default:
            return vscode.DiagnosticSeverity.Hint;
    }
}

function mapLanguageId(languageId: string): string {
    const mapping: Record<string, string> = {
        'python': 'python',
        'javascript': 'javascript',
        'typescript': 'typescript',
        'javascriptreact': 'javascript',
        'typescriptreact': 'typescript',
        'java': 'java',
        'php': 'php',
        'go': 'go',
        'ruby': 'ruby',
        'csharp': 'csharp'
    };
    return mapping[languageId] || languageId;
}

function isSupportedLanguage(languageId: string): boolean {
    const supported = ['python', 'javascript', 'typescript', 'javascriptreact', 
                       'typescriptreact', 'java', 'php', 'go', 'ruby', 'csharp'];
    return supported.includes(languageId);
}

function getSupportedLanguages(): vscode.DocumentSelector {
    return [
        { language: 'python' },
        { language: 'javascript' },
        { language: 'typescript' },
        { language: 'java' },
        { language: 'php' },
        { language: 'go' },
        { language: 'ruby' },
        { language: 'csharp' }
    ];
}

function openSettings() {
    vscode.commands.executeCommand('workbench.action.openSettings', 'aiVulnScanner');
}
