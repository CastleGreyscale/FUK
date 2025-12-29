/**
 * Placeholder Tab Component
 * Used for tabs that aren't implemented yet
 */

import Footer from '../components/Footer';

export default function PlaceholderTab({ 
  title, 
  description, 
  icon: Icon, 
  activeTab, 
  setActiveTab 
}) {
  return (
    <>
      <div className="fuk-preview-area">
        <div className="fuk-card-dashed" style={{ width: '40%' }}>
          <div className="fuk-placeholder">
            {Icon && <Icon style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.5 }} />}
            <p>{title}</p>
            {description && (
              <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                {description}
              </p>
            )}
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
        generateLabel="Not Available"
      />
    </>
  );
}
