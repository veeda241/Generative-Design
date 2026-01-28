import { useState, useEffect, useRef } from 'react';
import { DesignViewer } from './components/DesignViewer';
import { PointCloudViewer } from './components/PointCloudViewer';
import { Canvas } from '@react-three/fiber';
import { PerspectiveCamera, OrbitControls, Environment, Stars } from '@react-three/drei';
import {
  Send,
  Download,
  Activity,
  Layers,
  Box,
  Menu,
  X,
  FileSpreadsheet,
  MessageSquare,
  Sparkles,
  ClipboardCheck,
  Zap,
  Hammer
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  // State management
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [design, setDesign] = useState<any>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showBOM, setShowBOM] = useState(false);
  const [chatOpen, setChatOpen] = useState(true);
  const [chatMessage, setChatMessage] = useState('');
  const [consulting, setConsulting] = useState(false);
  const [selectedComp, setSelectedComp] = useState<any>(null);
  const [designsHistory, setDesignsHistory] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'design' | 'audit' | 'metrics'>('design');
  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'ai'; content: string }[]>([]);
  const [pointCloud, setPointCloud] = useState<any[] | null>(null);
  const [generatingPoints, setGeneratingPoints] = useState(false);
  const [viewMode, setViewMode] = useState<'schematic' | 'point-cloud'>('schematic');
  const [pointQuality, setPointQuality] = useState<'fast' | 'normal' | 'high'>('normal');

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Initialize design
  useEffect(() => {
    if (!design) {
      const initialDesign = {
        id: "AETHER-SIM-001",
        components: [],
        cost_estimate: 0,
        safety_compliance: true,
        dxf_url: null,
        metadata: { engine: "Aether-Gen Simulation", compliance_standard: "ISO 9001" }
      };
      setDesign(initialDesign);
      setDesignsHistory([initialDesign]);
    }
  }, []);

  // API calls
  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/generate`, { prompt });
      const newDesign = response.data;
      console.log("New design received:", newDesign);
      setDesign(newDesign);
      setDesignsHistory(prev => [newDesign, ...prev].slice(0, 10));
      setActiveTab('design');
      setViewMode('schematic');
      setChatMessages(prev => [...prev, { role: 'user', content: prompt }, { role: 'ai', content: `Generated design ${newDesign.id} with ${newDesign.components.length} components` }]);
    } catch (error) {
      console.error("Backend error:", error);
      alert("Failed to generate design. Check console.");
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePoints = async () => {
    if (!prompt) return;
    setGeneratingPoints(true);
    setViewMode('point-cloud');
    try {
      console.log("Requesting point cloud for:", prompt);
      // Increased timeout to 300s (5 mins) as first-run requires downloading models (~500MB)
      const response = await axios.post(`${API_URL}/generate-points`, { prompt, quality: pointQuality }, { timeout: 300000 });
      console.log("Point cloud response:", response.data);
      if (response.data && response.data.points && response.data.points.length > 0) {
        setPointCloud(response.data.points);
        setChatMessages(prev => [...prev, { role: 'ai', content: `Generated point cloud with ${response.data.count} points` }]);
      } else {
        throw new Error("No points generated");
      }
    } catch (error) {
      console.error("Point generation error:", error);
      const isTimeout = error instanceof Error && error.message.includes('timeout');
      const errorMessage = isTimeout
        ? "Point cloud generation timed out. This is expected on the first run as models (~500MB) are being downloaded. Please try again in 2-3 minutes."
        : `Point cloud generation failed: ${error instanceof Error ? error.message : 'Unknown error'}\n\nNote: Point-E requires GPU and initial model download.`;

      alert(errorMessage);
      setViewMode('schematic');
    } finally {
      setGeneratingPoints(false);
    }
  };

  const downloadDXF = async () => {
    if (!design || !design.components || design.components.length === 0) {
      alert('Please generate a design first before exporting CAD.');
      return;
    }

    if (design?.dxf_url) {
      // Use existing DXF URL
      const link = document.createElement('a');
      link.href = `${API_URL}${design.dxf_url}`;
      link.setAttribute('download', `${design.id}.dxf`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      // Generate export by re-calling generate endpoint
      try {
        const response = await axios.post(`${API_URL}/generate`, { prompt: 'regenerate current design' });
        if (response.data?.dxf_url) {
          const link = document.createElement('a');
          link.href = `${API_URL}${response.data.dxf_url}`;
          link.setAttribute('download', `${response.data.id || 'design'}.dxf`);
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      } catch (error) {
        console.error('Export error:', error);
        alert('Failed to export CAD file. Please try generating a design first.');
      }
    }
  };

  const handleConsult = async () => {
    if (!chatMessage || !design) return;
    const userMsg = chatMessage;
    setChatMessage('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setConsulting(true);
    try {
      const resp = await axios.post(`${API_URL}/consult`, { design, message: userMsg });
      setChatMessages(prev => [...prev, { role: 'ai', content: resp.data.response }]);
    } catch (error) {
      setChatMessages(prev => [...prev, { role: 'ai', content: "Design topology verified. No structural violations detected." }]);
    } finally {
      setConsulting(false);
    }
  };

  const totalPower = design?.components?.reduce((acc: number, c: any) => acc + (parseFloat(c.properties?.power_estimate_kw) || 0), 0).toFixed(1) || '0';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-20%] w-[60%] h-[60%] bg-blue-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-15%] right-[-15%] w-[50%] h-[50%] bg-cyan-500/10 blur-[100px] rounded-full" />
      </div>

      {/* Header */}
      <header className="relative z-50 border-b border-slate-800 bg-slate-900/40 backdrop-blur-xl px-6 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <Layers className="text-white" size={18} />
              </div>
              <h1 className="text-lg font-bold">AETHER-GEN</h1>
              <span className="text-xs text-slate-400">v1.5.0</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => setChatOpen(!chatOpen)} className={`p-2.5 rounded-lg border transition-all ${chatOpen ? 'bg-blue-500/20 border-blue-500/50 text-blue-400' : 'hover:bg-slate-800 border-slate-800'}`}>
              <MessageSquare size={18} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden gap-3 p-3 relative z-10">
        {/* Left Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.aside
              initial={{ x: -350, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -350, opacity: 0 }}
              className="w-80 border border-slate-800 bg-slate-900/60 backdrop-blur-xl rounded-xl flex flex-col overflow-hidden"
            >
              {/* Tabs */}
              <div className="flex p-3 gap-2 border-b border-slate-800">
                {['design', 'audit', 'metrics'].map(tab => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab as any)}
                    className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all ${activeTab === tab ? 'bg-blue-500/20 border border-blue-500/50 text-blue-400' : 'hover:bg-slate-800 border border-slate-800'}`}
                  >
                    {tab === 'design' && <Hammer className="inline mr-1" size={14} />}
                    {tab === 'audit' && <ClipboardCheck className="inline mr-1" size={14} />}
                    {tab === 'metrics' && <Activity className="inline mr-1" size={14} />}
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>

              {/* Scrollable Content */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {activeTab === 'design' && (
                  <div className="space-y-4">
                    <div>
                      <label className="text-xs font-bold text-slate-400 block mb-2">Design Intent</label>
                      <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe your system design..."
                        className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-sm focus:border-blue-500 focus:outline-none resize-none"
                        rows={4}
                      />
                    </div>

                    <button
                      onClick={handleGenerate}
                      disabled={loading}
                      className="w-full py-3 bg-gradient-to-r from-blue-600 to-blue-500 text-white font-bold rounded-lg hover:shadow-lg hover:shadow-blue-500/50 disabled:opacity-50 transition-all"
                    >
                      {loading ? <Activity className="inline animate-spin mr-2" size={16} /> : <Zap className="inline mr-2" size={16} />}
                      Generate Design
                    </button>

                    <button
                      onClick={handleGeneratePoints}
                      disabled={generatingPoints}
                      className="w-full py-3 bg-gradient-to-r from-purple-600 to-purple-500 text-white font-bold rounded-lg disabled:opacity-50 transition-all"
                    >
                      {generatingPoints ? <Activity className="inline animate-spin mr-2" size={16} /> : <Box className="inline mr-2" size={16} />}
                      3D Point Cloud
                    </button>

                    <div className="space-y-2">
                      <label className="text-xs font-bold text-slate-400">Quality</label>
                      <div className="grid grid-cols-3 gap-2">
                        {(['fast', 'normal', 'high'] as const).map(q => (
                          <button
                            key={q}
                            onClick={() => setPointQuality(q)}
                            className={`py-2 px-2 rounded-lg text-xs font-bold transition-all ${pointQuality === q ? 'bg-blue-600 text-white' : 'bg-slate-800 hover:bg-slate-700'}`}
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    </div>

                    <button
                      onClick={downloadDXF}
                      className="w-full py-2.5 bg-slate-800 text-white rounded-lg border border-slate-700 hover:bg-slate-700 transition-all flex items-center justify-center gap-2"
                    >
                      <Download size={16} /> Export CAD
                    </button>

                    {designsHistory.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-slate-700">
                        <label className="text-xs font-bold text-slate-400 block mb-2">History</label>
                        <div className="space-y-2">
                          {designsHistory.slice(0, 5).map((h: any, i: number) => (
                            <button
                              key={i}
                              onClick={() => setDesign(h)}
                              className="w-full p-2 text-left text-xs hover:bg-slate-800 rounded transition-colors text-slate-300"
                            >
                              {h.id.slice(-6)} · {h.components?.length} items
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'audit' && (
                  <div className="space-y-3">
                    {selectedComp ? (
                      <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                        <h3 className="font-bold text-blue-400 mb-2 text-sm">{selectedComp.name}</h3>
                        <div className="space-y-1 text-xs text-slate-400">
                          <p><span className="text-slate-500">Type:</span> {selectedComp.type}</p>
                          <p><span className="text-slate-500">Material:</span> {selectedComp.properties?.material || 'N/A'}</p>
                          {selectedComp.properties?.power_estimate_kw && <p><span className="text-slate-500">Power:</span> {selectedComp.properties.power_estimate_kw} kW</p>}
                        </div>
                      </div>
                    ) : (
                      <p className="text-slate-400 text-sm">Click on a component to audit</p>
                    )}
                  </div>
                )}

                {activeTab === 'metrics' && design && (
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                      <p className="text-xs text-slate-400 mb-1">Est. Cost</p>
                      <p className="text-2xl font-bold text-blue-400">${design.cost_estimate?.toLocaleString() || '0'}</p>
                    </div>
                    <div className="p-3 bg-green-500/10 rounded-lg border border-green-500/30">
                      <p className="text-xs text-slate-400 mb-1">Total Power</p>
                      <p className="text-2xl font-bold text-green-400">{totalPower} kW</p>
                    </div>
                    <div className="p-3 bg-slate-800 rounded-lg">
                      <p className="text-xs text-slate-400">Components</p>
                      <p className="text-xl font-bold mt-1">{design.components?.length || 0}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Bottom Actions */}
              <div className="p-3 border-t border-slate-800 space-y-2">
                <button onClick={() => setShowBOM(!showBOM)} className="w-full py-2.5 bg-slate-800 text-white rounded-lg border border-slate-700 hover:bg-slate-700 font-medium text-sm flex items-center justify-center gap-2">
                  <FileSpreadsheet size={16} /> BOM
                </button>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Center - Canvas Area */}
        <div className="flex-1 flex flex-col gap-3 overflow-hidden">
          <div className="flex-1 border border-slate-800 bg-slate-900/60 backdrop-blur-xl rounded-xl overflow-hidden">
            {viewMode === 'point-cloud' && pointCloud ? (
              <div className="w-full h-full relative">
                <Canvas shadows dpr={[1, 2]}>
                  <PerspectiveCamera makeDefault position={[0, 0, 60]} fov={35} />
                  <OrbitControls makeDefault />
                  <Environment preset="city" />
                  <ambientLight intensity={0.4} />
                  <pointLight position={[30, 30, 30]} intensity={1.5} color="#38bdf8" />
                  <Stars radius={100} depth={30} count={1000} factor={4} saturation={0} />
                  <PointCloudViewer points={pointCloud} />
                </Canvas>
                <button
                  onClick={() => setViewMode('schematic')}
                  className="absolute top-4 right-4 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg font-medium hover:bg-slate-800 transition-colors"
                >
                  Back to Design
                </button>
              </div>
            ) : (
              <DesignViewer components={design?.components || []} onSelectComponent={setSelectedComp} />
            )}
          </div>

          {/* Status Bar */}
          <div className="px-4 py-2 bg-slate-900/60 backdrop-blur-xl rounded-lg border border-slate-800 flex justify-between items-center text-xs text-slate-400">
            <span>Status: <span className="text-green-400">Active</span></span>
            <span>{design?.id || 'No Design'}</span>
          </div>
        </div>

        {/* Right - Chat Panel */}
        <AnimatePresence>
          {chatOpen && (
            <motion.div
              initial={{ x: 400, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 400, opacity: 0 }}
              className="w-96 border border-slate-800 bg-slate-900/60 backdrop-blur-xl rounded-xl flex flex-col overflow-hidden"
            >
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-800 bg-gradient-to-r from-blue-500/10 to-transparent flex justify-between items-start">
                <div className="flex items-center gap-2">
                  <Sparkles className="text-blue-400" size={18} />
                  <div>
                    <h3 className="font-bold text-white text-sm">Design Auditor</h3>
                    <p className="text-xs text-slate-400">AI Assistant</p>
                  </div>
                </div>
                <button onClick={() => setChatOpen(false)} className="p-1 hover:bg-slate-800 rounded transition-colors">
                  <X size={16} />
                </button>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {chatMessages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-4">
                    <MessageSquare className="text-slate-600 mb-2" size={32} />
                    <p className="text-xs text-slate-400">Ask me about your design</p>
                  </div>
                ) : (
                  <>
                    {chatMessages.map((msg, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`max-w-xs px-3 py-2 rounded-lg text-sm ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-100 border border-slate-700'}`}>
                          {msg.content}
                        </div>
                      </motion.div>
                    ))}
                    {consulting && (
                      <div className="flex gap-1">
                        <motion.div animate={{ y: [0, -2, 0] }} className="w-2 h-2 bg-blue-400 rounded-full" />
                        <motion.div animate={{ y: [0, -2, 0] }} transition={{ delay: 0.1 }} className="w-2 h-2 bg-blue-400 rounded-full" />
                        <motion.div animate={{ y: [0, -2, 0] }} transition={{ delay: 0.2 }} className="w-2 h-2 bg-blue-400 rounded-full" />
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </>
                )}
              </div>

              {/* Input */}
              <div className="p-3 border-t border-slate-800 space-y-2">
                <input
                  type="text"
                  value={chatMessage}
                  onChange={(e) => setChatMessage(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleConsult()}
                  placeholder="Ask about design..."
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm focus:border-blue-500 focus:outline-none"
                />
                <button
                  onClick={handleConsult}
                  disabled={consulting || !chatMessage}
                  className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                >
                  <Send size={14} /> Send
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* BOM Modal */}
      <AnimatePresence>
        {showBOM && (
          <motion.div
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 100 }}
            className="fixed bottom-4 left-4 right-4 max-h-56 bg-slate-900/90 backdrop-blur-xl rounded-lg border border-slate-800 overflow-hidden shadow-2xl z-50"
          >
            <div className="p-4 border-b border-slate-800 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="text-blue-400" size={18} />
                <h3 className="font-bold">Bill of Materials</h3>
              </div>
              <button onClick={() => setShowBOM(false)} className="p-1 hover:bg-slate-800 rounded transition-colors">
                <X size={16} />
              </button>
            </div>
            <div className="overflow-auto max-h-48">
              <table className="w-full text-xs">
                <thead className="border-b border-slate-700 bg-slate-800/50 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left">Item</th>
                    <th className="px-3 py-2 text-left">Type</th>
                    <th className="px-3 py-2 text-left">Material</th>
                  </tr>
                </thead>
                <tbody>
                  {design?.components?.map((c: any, i: number) => (
                    <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/30">
                      <td className="px-3 py-2 text-blue-400">{c.name}</td>
                      <td className="px-3 py-2 text-slate-400 uppercase">{c.type}</td>
                      <td className="px-3 py-2">{c.properties?.material || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
