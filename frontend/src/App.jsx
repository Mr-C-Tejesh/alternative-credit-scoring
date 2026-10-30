import { useState } from 'react'

const API = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const altFields = [
  ['utility_payment_timeliness', 'Utility payment timeliness', .05, 1, .82, .01],
  ['rental_payment_timeliness', 'Rental payment timeliness', .05, 1, .80, .01],
  ['digital_payment_regularity', 'Digital payment regularity', .02, 1, .75, .01],
  ['cashflow_stability_index', 'Cash-flow stability index', .02, 1, .68, .01],
  ['alternative_history_months', 'Alternative history (months)', 0, 24, 15, 1],
]

const initial = { age_years: 30, monthly_income_inr: 40000, employment_tenure_years: 4, debt_to_income_ratio: .35, existing_loan_count: 1, credit_utilization_ratio: .40, prior_delinquencies_24m: 0, bureau_history_available: true, bureau_score: 650, utility_payment_timeliness: .82, rental_payment_timeliness: .80, digital_payment_regularity: .75, cashflow_stability_index: .68, alternative_history_months: 15 }

function Field({ label, name, value, onChange, min, max, step = 1, disabled, hint }) {
  return <label className="field"><span>{label}</span><input name={name} type="number" value={value} min={min} max={max} step={step} disabled={disabled} onChange={onChange} required />{hint && <small>{hint}</small>}</label>
}

function Factor({ item, kind }) {
  return <article className={`factor ${kind}`}><strong>{item.readable_name}</strong><span>{Number(item.contribution).toFixed(3)} log-odds</span></article>
}

export default function App() {
  const [values, setValues] = useState(initial)
  const [unavailable, setUnavailable] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const update = event => setValues(current => ({ ...current, [event.target.name]: event.target.type === 'checkbox' ? event.target.checked : Number(event.target.value) }))
  const toggleMissing = name => setUnavailable(current => current.includes(name) ? current.filter(item => item !== name) : [...current, name])
  const submit = async event => {
    event.preventDefault(); setError(''); setResult(null); setLoading(true)
    const payload = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, key === 'bureau_history_available' ? Number(value) : value]))
    if (!values.bureau_history_available) payload.bureau_score = null
    unavailable.forEach(name => { payload[name] = null })
    try {
      const response = await fetch(`${API}/v1/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `The API returned HTTP ${response.status}.`)
      setResult(body)
    } catch (err) {
      setError(`No prediction was generated. ${err.message} Check VITE_API_BASE_URL and the API CORS allow-list.`)
    } finally { setLoading(false) }
  }
  const riskClass = result?.risk_band?.toLowerCase().startsWith('higher') ? 'risk-high' : result?.risk_band?.toLowerCase().startsWith('middle') ? 'risk-mid' : 'risk-low'
  return <main>
    <header className="hero"><p className="eyebrow">Alternative credit scoring · research interface</p><h1>Understand the model,<br />not just the number.</h1><p className="intro">A transparent interface for the project’s synthetic default-risk model.</p><span className="api-pill">● API connected to {API}</span></header>
    <aside className="notice"><strong>Educational use only.</strong> This prototype is trained on synthetic data. Do not enter real personal information or use outputs for lending, pricing, underwriting, or adverse-action decisions.</aside>
    <form onSubmit={submit}>
      <section className="panel"><div className="heading"><div><p className="eyebrow dark">01 · APPLICATION PROFILE</p><h2>Traditional financial signals</h2></div><p>Values are limited to the backend’s validated synthetic training range.</p></div>
        <div className="grid three"><Field label="Age (years)" name="age_years" min={20} max={65} value={values.age_years} onChange={update}/><Field label="Monthly income (₹)" name="monthly_income_inr" min={8000} max={250000} step={1000} value={values.monthly_income_inr} onChange={update}/><Field label="Employment tenure (years)" name="employment_tenure_years" min={0} max={35} step={.5} value={values.employment_tenure_years} onChange={update}/><Field label="Debt-to-income ratio" name="debt_to_income_ratio" min={0} max={1.2} step={.01} value={values.debt_to_income_ratio} onChange={update}/><Field label="Existing loan count" name="existing_loan_count" min={0} max={8} value={values.existing_loan_count} onChange={update}/><Field label="Credit utilization ratio" name="credit_utilization_ratio" min={0} max={1.25} step={.01} value={values.credit_utilization_ratio} onChange={update}/><Field label="Prior delinquencies (24 months)" name="prior_delinquencies_24m" min={0} max={5} value={values.prior_delinquencies_24m} onChange={update}/>
          <label className="toggle"><input name="bureau_history_available" type="checkbox" checked={values.bureau_history_available} onChange={update}/><span /><b>Bureau history available</b><small>Turn off to send a null bureau score.</small></label><Field label="Bureau score" name="bureau_score" min={300} max={900} step={5} value={values.bureau_score} onChange={update} disabled={!values.bureau_history_available}/></div>
      </section>
      <section className="panel"><div className="heading"><div><p className="eyebrow dark">02 · ALTERNATIVE DATA</p><h2>Behavioural summaries</h2></div><p>Tick an unavailable source to send `null`; the API supports missing alternative inputs.</p></div>
        <div className="grid three">{altFields.map(([name, label, min, max, defaultValue, step]) => <div className="alt" key={name}><Field label={label} name={name} min={min} max={max} step={step} value={values[name]} onChange={update} disabled={unavailable.includes(name)}/><button type="button" className={unavailable.includes(name) ? 'selected' : ''} onClick={() => toggleMissing(name)}>{unavailable.includes(name) ? 'Marked unavailable' : 'Mark unavailable'}</button></div>)}</div>
      </section>
      <button className="submit" disabled={loading}>{loading ? 'Running synthetic model…' : 'Run synthetic assessment'} <span>→</span></button>
    </form>
    {error && <div className="error" role="alert">{error}</div>}
    {result && <section className="results"><div className="heading"><div><p className="eyebrow dark">03 · MODEL OUTPUT</p><h2>Assessment result</h2></div><p>From the live `POST /v1/predict` response.</p></div>
      <div className="metrics"><article><span>Estimated default probability</span><strong>{(result.default_probability_12m * 100).toFixed(1)}%</strong><i>12-month synthetic estimate</i></article><article><span>Illustrative score</span><strong>{result.scoring.display_score}<em> / {result.scoring.max_score}</em></strong><i>PDO transformation</i></article><article className={riskClass}><span>Model-estimated risk band</span><strong>{result.risk_band}</strong><i>Model output, not a lending decision</i></article></div>
      <div className="explain"><div><h3>Why the model moved</h3><p>Exact local additive contributions from the hybrid logistic regression model. They describe model behaviour, not causes or legally compliant reason codes.</p></div><div className="factor-grid"><div><h4>↑ Increases risk</h4>{result.top_risk_increasing_factors.map(item => <Factor key={item.feature} item={item} kind="up"/>)}</div><div><h4>↓ Decreases risk</h4>{result.top_risk_decreasing_factors.map(item => <Factor key={item.feature} item={item} kind="down"/>)}</div></div></div>
      <aside className="disclaimer">{result.disclaimer}</aside>
    </section>}
  </main>
}
