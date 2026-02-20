import React, { useEffect, useRef, useState } from 'react';
import type { State } from '../types';
import { RefreshCw, Map as MapIcon, Grid, Bot, Play, Send, ChevronRight, ChevronDown } from 'lucide-react';

interface ChatPanelProps {
    state: State | null;
    onReset: () => void;
    onToggleMode: () => void;
    onPredict: (model: string, manualText?: string) => void;
    isPredicting: boolean;
}

// interface ContextBlockProps {
//     content: string;
// }

// const ContextBlock: React.FC<ContextBlockProps> = ({ content }) => {
//     // Basic parsing to split sections
//     // Assuming format like:
//     // ### Task Description
//     // ...
//     // ### Current Status
//     // ...
//     // ### Candidate Nodes
//     // ...
    
//     // We can split by "###" and reconstruct
//     const sections = content.split("###").filter(s => s.trim());
    
//     if (sections.length === 0) return <div className="whitespace-pre-wrap">{content}</div>;

//     return (
//         <div className="space-y-3">
//             {sections.map((sec, idx) => {
//                 const lines = sec.trim().split('\n');
//                 const title = lines[0].trim();
//                 const body = lines.slice(1).join('\n').trim();
                
//                 return (
//                     <div key={idx} className="bg-white/50 rounded-lg p-2 border border-amber-200/50">
//                         <div className="text-[10px] font-bold text-amber-700 uppercase mb-1 border-b border-amber-100 pb-1">
//                             {title}
//                         </div>
//                         <div className="whitespace-pre-wrap text-[10px] text-gray-600 leading-relaxed">
//                             {body}
//                         </div>
//                     </div>
//                 );
//             })}
//         </div>
//     );
// };

const MODELS = ["STS (Ours)", "Gemini 1.5 Pro", "GPT-4o", "GLM-4v", "Manual Input"];

const ChatPanel: React.FC<ChatPanelProps> = ({ state, onReset, onToggleMode, onPredict, isPredicting }) => {
    const logsEndRef = useRef<HTMLDivElement>(null);
    const [selectedModel, setSelectedModel] = useState(MODELS[0]);
    const [manualInput, setManualInput] = useState("");
    const [showPrompt, setShowPrompt] = useState(false);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [state?.logs]);

    const handleManualSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!manualInput.trim()) return;
        
        // Pass the manual input as the "model" for now, or handle differently.
        // For simplicity, we assume the parent component handles "Manual Input" logic if passed.
        // But the current API expects `model_name`.
        // We might need to handle this in `onPredict` or create a new prop.
        // Since `onPredict` takes a string, we can pass a special string or just "Manual Input" 
        // and let the App.tsx handle the parsing if needed. 
        // However, the prompt says "input text as action".
        // Let's assume we pass "Manual: <text>" to onPredict?
        // Or better, let's just use onPredict("Manual Input") and maybe have a way to pass the text?
        // Actually, let's update `onPredict` to optionally accept text.
        // But `onPredict` signature is `(model: string) => void`.
        // We should probably just parse the node ID here if we want to be quick, 
        // BUT the user might want to type "Go to node 5".
        // Let's change the pattern: If manual, we parse here and call `onPredict` with a special format?
        // No, let's stick to the request: "Manual option... input box... input text as action".
        
        // I'll assume the App.tsx `handlePredict` can handle "Manual Input" and maybe I need to pass the text somehow.
        // But `onPredict` only takes `model`. 
        // Let's modify the prop to `onPredict: (model: string, text?: string) => void` in the interface?
        // No, I can't change App.tsx easily without reading it again.
        // Wait, I can read App.tsx.
        // Let's just modify the interface here and I will update App.tsx in next step.
        
        onPredict(selectedModel, manualInput);
        setManualInput("");
    };

    // Helper to format time
    const formatTime = (seconds: number) => {
        const startHour = 15;
        const totalSeconds = Math.floor(seconds) + startHour * 3600;
        
        const h = Math.floor(totalSeconds / 3600) % 24;
        const m = Math.floor((totalSeconds % 3600) / 60);
        const s = totalSeconds % 60;
        
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    if (!state) return <div className="p-4 flex items-center justify-center h-full text-gray-500">Loading...</div>;

    return (
        <div className="flex flex-col h-full bg-white z-20">
            {/* Header */}
            <div className="px-6 py-5 border-b border-gray-100 bg-gradient-to-r from-white to-gray-50/50 backdrop-blur-sm sticky top-0 z-30">
                <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-bold text-gray-800 flex items-center">
                        <div className="p-1.5 bg-indigo-100 rounded-lg mr-2.5 text-indigo-600">
                            <Bot size={20} />
                        </div>
                        STS-VRP Controller
                    </h2>
                    <div className="flex space-x-1">
                         <div className={`w-2 h-2 rounded-full ${state.remaining_capacity > 0 ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
                    </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-100 rounded-lg p-2.5 flex flex-col items-center justify-center">
                        <span className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-0.5">Total Cost</span>
                        <span className="text-lg font-mono font-bold text-gray-800 tracking-tight">{state.current_cost.toFixed(2)}</span>
                    </div>
                    <div className="bg-gray-50 rounded-2xl p-3 flex flex-col items-center justify-center border border-gray-100">
                        <span className="text-[10px] text-gray-400 uppercase tracking-widest font-bold mb-1">Current Time</span>
                        <div className="flex items-end">
                            <span className="text-lg font-mono font-bold tracking-tight text-gray-800">
                                {formatTime(state.current_cost)}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Environment Prompt (Collapsible) */}
            {state.text_prompt && (
                <div className="px-6 py-2 bg-gray-50/50 border-b border-gray-100 transition-all">
                    <button 
                        onClick={() => setShowPrompt(!showPrompt)}
                        className="w-full flex items-center justify-between text-[10px] font-bold text-gray-400 uppercase tracking-widest hover:text-indigo-600 transition-colors py-1"
                    >
                        <span className="flex items-center">
                            {showPrompt ? <ChevronDown size={14} className="mr-1" /> : <ChevronRight size={14} className="mr-1" />}
                            Current Environment Prompt (Context)
                        </span>
                        <span className="text-[9px] bg-gray-100 px-1.5 py-0.5 rounded text-gray-500">
                            {state.text_prompt.length} chars
                        </span>
                    </button>
                    
                    {showPrompt && (
                        <div className="mt-2 p-3 bg-white border border-gray-200 rounded-xl text-[10px] font-mono text-gray-600 whitespace-pre-wrap max-h-48 overflow-y-auto shadow-inner animate-in fade-in slide-in-from-top-1 duration-200">
                            {state.text_prompt}
                        </div>
                    )}
                </div>
            )}

            {/* Chat/Logs Area */}
            <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-white scroll-smooth">
                {state.logs.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-gray-300 space-y-4">
                        <div className="p-4 bg-gray-50 rounded-full">
                            <Bot size={32} />
                        </div>
                        <span className="text-sm font-medium">Ready to start mission</span>
                    </div>
                )}
                {state.logs.map((log, idx) => {
                    // Skip context logs
                    if (log.startsWith("[Context Update]")) return null;
                    
                    const isModel = log.startsWith("Model");
                    
                    return (
                        <div key={idx} className={`flex ${isModel ? 'justify-start' : 'justify-end'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                            <div className={`rounded-2xl p-4 text-sm leading-relaxed shadow-sm ${
                                isModel 
                                    ? 'bg-gray-50 text-gray-700 rounded-tl-sm border border-gray-100 max-w-[85%]' 
                                    : 'bg-indigo-600 text-white rounded-tr-sm shadow-indigo-100 max-w-[85%]'
                            }`}>
                                <div className={`font-bold text-[10px] uppercase tracking-wider mb-1.5 ${
                                    isModel 
                                        ? 'text-indigo-500' 
                                        : 'text-indigo-200'
                                }`}>
                                    {isModel ? "STS-VRP Agent" : "Operator"}
                                </div>
                                <div className="whitespace-pre-wrap font-medium">{log}</div>
                            </div>
                        </div>
                    );
                })}
                {isPredicting && (
                    <div className="flex justify-start">
                         <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-none p-4 shadow-sm flex items-center space-x-2">
                            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                    </div>
                )}
                <div ref={logsEndRef} />
            </div>

            {/* Controls */}
            <div className="p-4 bg-white border-t border-gray-100 space-y-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-10">
                
                {/* Model Selector & Input */}
                <div className="space-y-3">
                    <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest ml-1">Control Interface</label>
                    <div className="flex space-x-2">
                        <div className="relative flex-1 group">
                            <select 
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full appearance-none bg-gray-50 border border-gray-200 text-gray-700 py-3 px-4 pr-8 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition text-sm font-medium hover:border-gray-300"
                            >
                                {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-gray-400">
                                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
                            </div>
                        </div>
                        
                        {selectedModel !== "Manual Input" && (
                            <button
                                onClick={() => onPredict(selectedModel)}
                                disabled={isPredicting || state.remaining_capacity <= 0}
                                className="flex-shrink-0 bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white p-3 rounded-2xl transition-all shadow-lg shadow-indigo-200 disabled:bg-gray-200 disabled:shadow-none disabled:cursor-not-allowed flex items-center justify-center min-w-[3.5rem]"
                            >
                                {isPredicting ? (
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                ) : (
                                    <Play size={20} fill="currentColor" className="ml-0.5" />
                                )}
                            </button>
                        )}
                    </div>

                    {selectedModel === "Manual Input" && (
                        <form onSubmit={handleManualSubmit} className="flex space-x-2 animate-in fade-in slide-in-from-top-1 duration-200">
                            <input
                                type="text"
                                value={manualInput}
                                onChange={(e) => setManualInput(e.target.value)}
                                placeholder="Type action (e.g., 'Go to Node 5')"
                                className="flex-1 bg-gray-50 border border-gray-200 text-gray-900 text-sm rounded-2xl focus:ring-indigo-500 focus:border-indigo-500 block p-3 outline-none transition-all hover:border-gray-300"
                                autoFocus
                                list="node-options"
                            />
                            <datalist id="node-options">
                                {state.nodes.map(n => {
                                    if (state.current_path.includes(n.id) || n.type === 'depot') return null;
                                    const start = n.time_window ? formatTime(n.time_window[0]) : "";
                                    const end = n.time_window ? formatTime(n.time_window[1]) : "";
                                    const tw = n.time_window ? `[${start} - ${end}]` : "";
                                    return (
                                        <option key={n.id} value={`Go to Node ${n.id}`}>
                                            {tw} Demand: {n.demand}
                                        </option>
                                    );
                                })}
                            </datalist>
                            <button
                                type="submit"
                                disabled={!manualInput.trim()}
                                className="bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white p-3 rounded-2xl transition-all shadow-lg shadow-indigo-200 disabled:bg-gray-200 disabled:shadow-none disabled:cursor-not-allowed flex items-center justify-center min-w-[3.5rem]"
                            >
                                <Send size={18} className="ml-0.5" />
                            </button>
                        </form>
                    )}
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                    <button
                        onClick={onReset}
                        className="flex items-center justify-center px-4 py-3 bg-rose-50 text-rose-600 border border-rose-100 rounded-2xl hover:bg-rose-100 hover:border-rose-200 transition-all text-sm font-semibold group"
                    >
                        <RefreshCw size={16} className="mr-2 group-hover:rotate-180 transition-transform duration-500" />
                        Reset
                    </button>
                    
                    <button
                        onClick={onToggleMode}
                        className="flex items-center justify-center px-4 py-3 bg-gray-50 text-gray-600 border border-gray-200 rounded-2xl hover:bg-gray-100 hover:border-gray-300 transition-all text-sm font-semibold"
                    >
                        {state.mode === 'real' ? (
                            <>
                                <Grid size={16} className="mr-2 text-gray-400" /> Virtual View
                            </>
                        ) : (
                            <>
                                <MapIcon size={16} className="mr-2 text-gray-400" /> Real Map
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatPanel;
