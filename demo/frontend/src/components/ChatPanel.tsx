import React, { useEffect, useRef, useState } from 'react';
import type { State } from '../types';
import { RefreshCw, Map as MapIcon, Grid, Bot, Play, Send } from 'lucide-react';

interface ChatPanelProps {
    state: State | null;
    onReset: () => void;
    onToggleMode: () => void;
    onPredict: (model: string, manualText?: string) => void;
    isPredicting: boolean;
}

const MODELS = ["STS (Ours)", "Gemini 1.5 Pro", "GPT-4o", "GLM-4v", "Manual Input"];

const ChatPanel: React.FC<ChatPanelProps> = ({ state, onReset, onToggleMode, onPredict, isPredicting }) => {
    const logsEndRef = useRef<HTMLDivElement>(null);
    const [selectedModel, setSelectedModel] = useState(MODELS[0]);
    const [manualInput, setManualInput] = useState("");

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

    if (!state) return <div className="p-4 flex items-center justify-center h-full text-gray-500">Loading...</div>;

    return (
        <div className="flex flex-col h-full bg-white border-l border-gray-200 shadow-2xl z-20">
            {/* Header */}
            <div className="px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-white to-gray-50">
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
                        <span className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-0.5">Current Cost</span>
                        <span className="text-lg font-mono font-bold text-gray-800">{state.current_cost.toFixed(2)}</span>
                    </div>
                    <div className="bg-gray-100 rounded-lg p-2.5 flex flex-col items-center justify-center">
                        <span className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-0.5">Capacity</span>
                        <div className="flex items-end">
                            <span className={`text-lg font-mono font-bold ${state.remaining_capacity < 5 ? 'text-red-600' : 'text-gray-800'}`}>
                                {state.remaining_capacity}
                            </span>
                            <span className="text-xs text-gray-400 mb-1 ml-1">/ {state.capacity}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Chat/Logs Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50 scroll-smooth">
                {state.logs.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-2 opacity-60">
                        <Bot size={48} />
                        <span className="text-sm">No activity yet. Start by selecting a node.</span>
                    </div>
                )}
                {state.logs.map((log, idx) => {
                    const isModel = log.startsWith("Model");
                    // const isUser = log.startsWith("User");
                    return (
                        <div key={idx} className={`flex ${isModel ? 'justify-start' : 'justify-end'}`}>
                            <div className={`max-w-[85%] rounded-2xl p-3.5 text-sm shadow-sm ${
                                isModel 
                                    ? 'bg-white border border-gray-100 text-gray-700 rounded-tl-none' 
                                    : 'bg-indigo-600 text-white rounded-tr-none'
                            }`}>
                                <div className={`font-semibold text-xs mb-1 opacity-75 ${isModel ? 'text-indigo-600' : 'text-indigo-100'}`}>
                                    {isModel ? "🤖 Model Agent" : "👤 User"}
                                </div>
                                <div className="leading-relaxed whitespace-pre-wrap">{log}</div>
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
                    <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider ml-1">Control Interface</label>
                    <div className="flex space-x-2">
                        <div className="relative flex-1">
                            <select 
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full appearance-none bg-gray-50 border border-gray-200 text-gray-700 py-2.5 px-4 pr-8 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition text-sm font-medium"
                            >
                                {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
                                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
                            </div>
                        </div>
                        
                        {selectedModel !== "Manual Input" && (
                            <button
                                onClick={() => onPredict(selectedModel)}
                                disabled={isPredicting || state.remaining_capacity <= 0}
                                className="flex-shrink-0 bg-indigo-600 hover:bg-indigo-700 text-white p-2.5 rounded-xl transition shadow-md shadow-indigo-200 disabled:bg-gray-300 disabled:shadow-none disabled:cursor-not-allowed flex items-center justify-center min-w-[3rem]"
                            >
                                {isPredicting ? (
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                ) : (
                                    <Play size={20} fill="currentColor" />
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
                                className="flex-1 bg-white border border-gray-300 text-gray-900 text-sm rounded-xl focus:ring-indigo-500 focus:border-indigo-500 block p-2.5 outline-none shadow-sm"
                                autoFocus
                            />
                            <button
                                type="submit"
                                disabled={!manualInput.trim()}
                                className="bg-indigo-600 hover:bg-indigo-700 text-white p-2.5 rounded-xl transition shadow-md shadow-indigo-200 disabled:bg-gray-300 disabled:shadow-none disabled:cursor-not-allowed flex items-center justify-center min-w-[3rem]"
                            >
                                <Send size={18} />
                            </button>
                        </form>
                    )}
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2">
                    <button
                        onClick={onReset}
                        className="flex items-center justify-center px-4 py-2.5 bg-red-50 text-red-600 border border-red-100 rounded-xl hover:bg-red-100 hover:border-red-200 transition text-sm font-medium group"
                    >
                        <RefreshCw size={16} className="mr-2 group-hover:rotate-180 transition-transform duration-500" />
                        Reset Env
                    </button>
                    
                    <button
                        onClick={onToggleMode}
                        className="flex items-center justify-center px-4 py-2.5 bg-gray-50 text-gray-700 border border-gray-200 rounded-xl hover:bg-gray-100 hover:border-gray-300 transition text-sm font-medium"
                    >
                        {state.mode === 'real' ? (
                            <>
                                <Grid size={16} className="mr-2 text-gray-500" /> Virtual View
                            </>
                        ) : (
                            <>
                                <MapIcon size={16} className="mr-2 text-gray-500" /> Real Map
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatPanel;
