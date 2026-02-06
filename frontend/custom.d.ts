// Global declaration for file types
declare module '*.png'
declare module '*.jpg'
declare module '*.svg'

// TypeScript Transpiler for Client-side
declare module 'typescript' {
    export const transpileModule: any;
    export const ScriptTarget: any;
    export const ModuleKind: any;
}
