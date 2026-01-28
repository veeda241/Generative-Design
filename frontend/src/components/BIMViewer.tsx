import { useEffect, useRef, useState } from 'react';
import { FileUp, Eye, Settings, RotateCcw } from 'lucide-react';

interface BIMViewerProps {
  onLoad?: (data: any) => void;
  className?: string;
}

export const BIMViewer: React.FC<BIMViewerProps> = ({ onLoad, className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: string } | null>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    // Initialize xeokit viewer
    if (containerRef.current && !viewerRef.current) {
      try {
        // Note: xeokit SDK would be initialized here
        console.log('BIM Viewer initialized');
      } catch (error) {
        console.error('Failed to initialize BIM viewer:', error);
      }
    }
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setFileInfo({
      name: file.name,
      size: (file.size / 1024 / 1024).toFixed(2) + ' MB'
    });

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/upload-ifc`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        onLoad?.(data);
        console.log('BIM file loaded:', data);
      }
    } catch (error) {
      console.error('BIM file upload error:', error);
      alert('Failed to load BIM file');
    } finally {
      setIsLoading(false);
    }
  };

  const resetView = () => {
    if (viewerRef.current) {
      viewerRef.current.scene.setAllObjectsVisible();
      viewerRef.current.scene.setAllObjectsXRayed(false);
      console.log('View reset');
    }
  };

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* Controls */}
      <div className="flex gap-2">
        <label className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer font-medium">
          <FileUp size={18} />
          {isLoading ? 'Loading IFC...' : 'Load IFC File'}
          <input
            type="file"
            accept=".ifc,.IFC"
            onChange={handleFileUpload}
            disabled={isLoading}
            className="hidden"
          />
        </label>
        <button
          onClick={resetView}
          className="px-4 py-2.5 bg-slate-800 text-slate-300 rounded-lg hover:bg-slate-700 transition-colors flex items-center gap-2"
          title="Reset view"
        >
          <RotateCcw size={18} />
        </button>
        <button
          className="px-4 py-2.5 bg-slate-800 text-slate-300 rounded-lg hover:bg-slate-700 transition-colors flex items-center gap-2"
          title="Viewer settings"
        >
          <Settings size={18} />
        </button>
      </div>

      {/* File Info */}
      {fileInfo && (
        <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
          <div className="flex items-center gap-2">
            <Eye className="text-blue-400" size={16} />
            <div className="text-sm">
              <p className="text-white font-medium">{fileInfo.name}</p>
              <p className="text-xs text-slate-400">{fileInfo.size}</p>
            </div>
          </div>
        </div>
      )}

      {/* Viewer Container */}
      <div
        ref={containerRef}
        className="flex-1 bg-slate-950 border border-slate-800 rounded-lg overflow-hidden min-h-[400px]"
        style={{
          perspective: '1000px'
        }}
      >
        <div className="w-full h-full flex items-center justify-center text-slate-400">
          <div className="text-center">
            <FileUp className="mx-auto mb-3 opacity-50" size={48} />
            <p className="text-sm">Upload an IFC file to view BIM model</p>
            <p className="text-xs text-slate-500 mt-2">Supported: IFC 2x3, IFC 4.0, IFC 4.1</p>
          </div>
        </div>
      </div>

      {/* Stats */}
      {fileInfo && (
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 bg-slate-800 rounded border border-slate-700">
            <p className="text-slate-400">Model Type</p>
            <p className="text-white font-mono">IFC</p>
          </div>
          <div className="p-2 bg-slate-800 rounded border border-slate-700">
            <p className="text-slate-400">Size</p>
            <p className="text-white font-mono">{fileInfo.size}</p>
          </div>
          <div className="p-2 bg-slate-800 rounded border border-slate-700">
            <p className="text-slate-400">Status</p>
            <p className="text-green-400 font-mono">Loaded</p>
          </div>
        </div>
      )}
    </div>
  );
};
