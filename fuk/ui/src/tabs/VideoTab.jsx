/**
 * Video Generation Tab (Placeholder)
 */

import { Film } from '../components/Icons';
import Footer from '../components/Footer';

export default function VideoTab({ config, activeTab, setActiveTab }) {
  return (
    <>
      <div className="fuk-preview-area">
        <div className="fuk-card-dashed" style={{ width: '40%' }}>
          <div className="fuk-placeholder">
            <Film style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.5 }} />
            <p>Video generation UI coming soon...</p>
            <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
              Wan I2V • FLF2V • Fun Control
            </p>
          </div>
        </div>
      </div>

      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={false}
        progress={null}
        elapsedSeconds={0}
        onGenerate={() => {}}
        onCancel={() => {}}
        canGenerate={false}
        generateLabel="Generate Video"
      />
    </>
  );
}
