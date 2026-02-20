import { useState, useEffect } from 'react';
import MapPanel from './components/MapPanel';
import ChatPanel from './components/ChatPanel';
import { getState, resetEnv, step, predict } from './api';
import type { State } from './types';

function App() {
    const [state, setState] = useState<State | null>(null);
    const [viewMode, setViewMode] = useState<"real" | "virtual">("real");
    const [isPredicting, setIsPredicting] = useState(false);

    // Initial load
    useEffect(() => {
        fetchState();
    }, []);

    const fetchState = async () => {
        try {
            const data = await getState();
            setState(data);
        } catch (error) {
            console.error("Failed to fetch state:", error);
            // If backend is fresh, it might be empty, so try reset
            try {
                const data = await resetEnv();
                setState(data);
            } catch (e) {
                console.error("Failed to reset:", e);
            }
        }
    };

    const handleNodeClick = async (nodeId: number) => {
        if (!state) return;
        try {
            const newState = await step(nodeId);
            setState(newState);
        } catch (error) {
            console.error("Failed to step:", error);
            alert("Error: " + (error as any).message);
        }
    };

    const handleReset = async () => {
        try {
            const newState = await resetEnv();
            setState(newState);
        } catch (error) {
            console.error("Failed to reset:", error);
        }
    };

    const handleToggleMode = () => {
        setViewMode(prev => prev === 'real' ? 'virtual' : 'real');
    };

    const handlePredict = async (model: string, manualText?: string) => {
        if (!state) return;
        setIsPredicting(true);
        try {
            if (model === "Manual Input" && manualText) {
                // Try to parse node ID from text
                // Regex to find "Node X" or just "X"
                const match = manualText.match(/(\d+)/);
                if (match) {
                    const nodeId = parseInt(match[1]);
                    // Check if node exists
                    const node = state.nodes.find(n => n.id === nodeId);
                    if (node) {
                         await handleNodeClick(nodeId);
                    } else {
                        alert(`Node ${nodeId} not found.`);
                    }
                } else {
                    alert("Could not understand the command. Please specify a node ID.");
                }
            } else {
                const res = await predict(model);
                await handleNodeClick(res.node_id);
            }
        } catch (error) {
            console.error("Prediction failed:", error);
            alert("Prediction failed");
        } finally {
            setIsPredicting(false);
        }
    };

    return (
        <div className="flex h-screen w-screen overflow-hidden bg-gray-50">
            {/* Left Panel: Map (60%) - Added rounded corners and margin for floating effect */}
            <div className="flex-1 h-full p-4 pr-2">
                <div className="h-full w-full relative rounded-3xl overflow-hidden shadow-2xl border border-gray-200 bg-white">
                    <MapPanel 
                        state={state} 
                        mode={viewMode} 
                        onNodeClick={handleNodeClick} 
                    />
                </div>
            </div>

            {/* Right Panel: Chat/Control (40%) - Floating card style */}
            <div className="w-[450px] h-full p-4 pl-2 flex flex-col">
                <div className="flex-1 h-full rounded-3xl overflow-hidden shadow-2xl border border-gray-200 bg-white flex flex-col">
                    <ChatPanel 
                        state={state ? { ...state, mode: viewMode } : null} 
                        onReset={handleReset}
                        onToggleMode={handleToggleMode}
                        onPredict={handlePredict}
                        isPredicting={isPredicting}
                    />
                </div>
            </div>
        </div>
    );
}

export default App;
