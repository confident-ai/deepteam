import os
import asyncio
from deepeval.models import GPTModel

from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
    IllegalActivity, Bias, Toxicity, PersonalSafety, 
    GraphicContent, Misinformation, ExcessiveAgency, 
    UnauthorizedAccess, PIILeakage, PromptLeakage
)
from deepteam.attacks.multi_turn import BadLikertJudge

def check_api_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    return True

# Setup target model using deepeval's GPTModel
target_model = None
if check_api_key():
    target_model = GPTModel(model="gpt-4o-mini")

async def model_callback(input: str) -> str:
    """Model callback for red teaming - simulates target model being attacked"""
    if not target_model:
        return "Error: OpenAI API key not configured"
    
    try:
        # Use deepeval's GPTModel which handles async automatically
        response = await target_model.a_generate(input)
        # Ensure we return a string, handle different response types
        if isinstance(response, tuple):
            return str(response[0]) if response else "Empty response"
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    """Test Bad Likert Judge attack across multiple vulnerability types using RedTeamer"""
    if not target_model:
        return
        
    print("🚀 Testing Bad Likert Judge Attack Across All Vulnerability Types...")
    print("=" * 80)
    
    # Create diverse vulnerability instances to test BLJ adaptability
    vulnerabilities = [
        IllegalActivity(types=["weapons", "illegal drugs"]),
        Bias(types=["religion", "politics"]),
        Toxicity(types=["threats", "insults"]),
        PersonalSafety(types=["self-harm", "bullying"]),
        GraphicContent(types=["sexual content"]),
        Misinformation(types=["factual errors"]),
        ExcessiveAgency(types=["functionality", "permissions"]),
        UnauthorizedAccess(types=["sql injection", "bfla"]),
        PIILeakage(types=["direct disclosure"]),
        PromptLeakage(types=["instructions", "secrets and credentials"])
    ]
    
    # Create Bad Likert Judge attack variants with different configurations
    attacks = [
        # Test core vulnerability categories with different configurations
        BadLikertJudge(category="illegal_activity", turns=3, enable_refinement=True),
        BadLikertJudge(category="bias", turns=2, enable_refinement=False),
        BadLikertJudge(category="toxicity", turns=3, enable_refinement=True),
        BadLikertJudge(category="personal_safety", turns=2, enable_refinement=False),
        BadLikertJudge(category="graphic_content", turns=3, enable_refinement=True),
        BadLikertJudge(category="misinformation", turns=2, enable_refinement=False),
        BadLikertJudge(category="excessive_agency", turns=3, enable_refinement=True),
        BadLikertJudge(category="unauthorized_access", turns=2, enable_refinement=False),
        BadLikertJudge(category="pii_leakage", turns=3, enable_refinement=True),
        BadLikertJudge(category="prompt_leakage", turns=2, enable_refinement=False),
        
        # Test specific vulnerability subtypes
        BadLikertJudge(category="weapons", turns=3, enable_refinement=True),
        BadLikertJudge(category="religion", turns=2, enable_refinement=False),
        BadLikertJudge(category="threats", turns=3, enable_refinement=True),
        BadLikertJudge(category="self-harm", turns=2, enable_refinement=False),
        BadLikertJudge(category="sql_injection", turns=3, enable_refinement=True),
        
        # Test with malware/hacking categories (should map to unauthorized_access pattern)
        BadLikertJudge(category="malware", turns=3, enable_refinement=True),
        BadLikertJudge(category="hacking", turns=2, enable_refinement=False),
        
        # Test edge cases and custom categories
        BadLikertJudge(category="violence", turns=3, enable_refinement=True),
        BadLikertJudge(category="harassment", turns=2, enable_refinement=False),
        BadLikertJudge(category="custom_harmful_category", turns=2, enable_refinement=False),
    ]
    
    print(f"📋 Created {len(attacks)} Bad Likert Judge attack variants:")
    for i, attack in enumerate(attacks, 1):
        try:
            name = attack.get_name()
            config = f"({attack.turns} turns, refinement: {'on' if attack.enable_refinement else 'off'})"
            print(f"  {i}. {name} {config}")
        except Exception as e:
            print(f"  {i}. [Error getting name: {str(e)}]")
    print()
    
    # Validate all attacks were created successfully
    valid_attacks = []
    for attack in attacks:
        try:
            # Test that we can get the name (basic validation)
            attack.get_name()
            valid_attacks.append(attack)
        except Exception as e:
            print(f"⚠️  Skipping invalid attack: {str(e)}")
    
    if not valid_attacks:
        print("❌ No valid attacks created! Cannot proceed.")
        return
        
    print(f"✅ Using {len(valid_attacks)} valid attacks out of {len(attacks)} created.")
    print(f"🎯 Testing against {len(vulnerabilities)} vulnerability types.")
    print()
    
    # Create RedTeamer instance
    red_teamer = RedTeamer(
        target_purpose="General AI assistant",
        async_mode=True,
        max_concurrent=2  # Conservative concurrency
    )
    
    # Temporarily monkey-patch the print method to avoid the error
    original_print_method = red_teamer._print_risk_assessment
    red_teamer._print_risk_assessment = lambda: print("📋 Risk assessment completed (printing bypassed)")
    
    try:
        # Run red team attack with all variants - TEST EACH ATTACK/VULNERABILITY COMBINATION
        print("🔥 Running Bad Likert Judge red team attacks...")
        
        all_test_cases = []
        attack_results = {}
        vulnerability_results = {}
        
        # Test a subset of attacks against vulnerabilities for comprehensive coverage
        # Take top 10 attack variants to avoid overwhelming the API
        test_attacks = valid_attacks[:10]
        test_vulnerabilities = vulnerabilities[:5]  # Test against 5 vulnerability types
        
        print(f"🎯 Testing {len(test_attacks)} attack variants against {len(test_vulnerabilities)} vulnerability types")
        
        for i, attack in enumerate(test_attacks, 1):
            attack_name = attack.get_name()
            print(f"\n🎯 Testing Attack {i}/{len(test_attacks)}: {attack_name}")
            
            try:
                # Run red team for this specific attack against multiple vulnerabilities
                risk_assessment = red_teamer.red_team(
                    model_callback=model_callback,
                    vulnerabilities=test_vulnerabilities,
                    attacks=[attack],  # Only this one attack
                    attacks_per_vulnerability_type=1,  # 1 per vulnerability type
                    ignore_errors=True
                )
                
                # Collect results
                for test_case in risk_assessment.test_cases:
                    all_test_cases.append(test_case)
                    
                    # Group by attack method
                    attack_method = test_case.attack_method or attack_name
                    if attack_method not in attack_results:
                        attack_results[attack_method] = []
                    attack_results[attack_method].append(test_case)
                    
                    # Group by vulnerability type
                    vuln_type = test_case.vulnerability_type
                    if vuln_type not in vulnerability_results:
                        vulnerability_results[vuln_type] = []
                    vulnerability_results[vuln_type].append(test_case)
                
                print(f"   ✅ Completed: {len(risk_assessment.test_cases)} test cases")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        print(f"\n📊 Bad Likert Judge Red Team Results:")
        print(f"Total test cases: {len(all_test_cases)}")
        print("=" * 80)
        
        # Debug: Show what we got
        if not all_test_cases:
            print("⚠️  No test cases generated! This might indicate an issue with attack execution.")
            return
        
        # Display results grouped by attack type
        print("\n🎯 RESULTS BY ATTACK VARIANT:")
        print("=" * 50)
        for attack_method, test_cases in attack_results.items():
            print(f"\n🔍 {attack_method}")
            print("-" * 60)
            
            successful = sum(1 for tc in test_cases if tc.score == 0)
            total = len(test_cases)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            print(f"Success Rate: {successful}/{total} ({success_rate:.1f}%)")
            
            # Group by vulnerability for this attack
            vuln_breakdown = {}
            for tc in test_cases:
                vuln = tc.vulnerability_type
                if vuln not in vuln_breakdown:
                    vuln_breakdown[vuln] = {'success': 0, 'total': 0}
                vuln_breakdown[vuln]['total'] += 1
                if tc.score == 0:
                    vuln_breakdown[vuln]['success'] += 1
            
            print("  Breakdown by vulnerability:")
            for vuln, stats in vuln_breakdown.items():
                rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"    {vuln}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
            
            # Show sample test case details
            if test_cases:
                print("\n  Sample Test Case:")
                sample_tc = test_cases[0]
                print(f"    Vulnerability: {sample_tc.vulnerability_type}")
                print(f"    Score: {sample_tc.score} ({'✅ Attack Succeeded' if sample_tc.score == 0 else '❌ Attack Mitigated'})")
                
                # Show a preview of the enhanced prompt
                if sample_tc.input:
                    preview = sample_tc.input.replace('\n', ' ')[:300] + "..."
                    print(f"    Enhanced Prompt: {preview}")
                
                # Show response preview if available
                if sample_tc.actual_output:
                    try:
                        if isinstance(sample_tc.actual_output, tuple):
                            response_text = str(sample_tc.actual_output[0]) if sample_tc.actual_output else "Empty tuple response"
                        elif isinstance(sample_tc.actual_output, str):
                            response_text = sample_tc.actual_output
                        else:
                            response_text = str(sample_tc.actual_output)
                        
                        response_preview = response_text.replace('\n', ' ')[:200] + "..."
                        print(f"    Model Response: {response_preview}")
                    except Exception as e:
                        print(f"    Model Response: [Error processing response: {str(e)}]")
                
                if sample_tc.reason:
                    print(f"    Evaluation: {sample_tc.reason}")
        
        # Display results grouped by vulnerability type
        print(f"\n🎯 RESULTS BY VULNERABILITY TYPE:")
        print("=" * 50)
        for vuln_type, test_cases in vulnerability_results.items():
            print(f"\n🔍 {vuln_type}")
            print("-" * 60)
            
            successful = sum(1 for tc in test_cases if tc.score == 0)
            total = len(test_cases)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            print(f"Success Rate: {successful}/{total} ({success_rate:.1f}%)")
            
            # Show which attack methods were most effective for this vulnerability
            attack_breakdown = {}
            for tc in test_cases:
                attack = tc.attack_method or "Unknown"
                if attack not in attack_breakdown:
                    attack_breakdown[attack] = {'success': 0, 'total': 0}
                attack_breakdown[attack]['total'] += 1
                if tc.score == 0:
                    attack_breakdown[attack]['success'] += 1
            
            print("  Most effective attack variants:")
            sorted_attacks = sorted(attack_breakdown.items(), 
                                  key=lambda x: (x[1]['success']/x[1]['total'] if x[1]['total'] > 0 else 0), 
                                  reverse=True)
            for attack, stats in sorted_attacks[:3]:  # Top 3
                rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                attack_short = attack.replace("Bad Likert Judge", "BLJ")
                print(f"    {attack_short}: {rate:.1f}% ({stats['success']}/{stats['total']})")
        
        # Overall summary
        total_tests = len(all_test_cases)
        total_successful = sum(1 for tc in all_test_cases if tc.score == 0)
        
        print(f"\n📈 OVERALL SUMMARY:")
        print("=" * 80)
        print(f"Total attack attempts: {total_tests}")
        print(f"Successful attacks: {total_successful}")
        print(f"Overall success rate: {total_successful/total_tests*100:.1f}%")
        
        # Attack configuration analysis
        config_analysis = {'3_turns_refined': [], '3_turns_basic': [], '2_turns_basic': []}
        for attack_method, test_cases in attack_results.items():
            if "3 turns, refinement: on" in str(attack_method):
                config_analysis['3_turns_refined'].extend(test_cases)
            elif "3 turns, refinement: off" in str(attack_method):
                config_analysis['3_turns_basic'].extend(test_cases)
            elif "2 turns" in str(attack_method):
                config_analysis['2_turns_basic'].extend(test_cases)
        
        print(f"\n🔧 CONFIGURATION ANALYSIS:")
        for config_name, test_cases in config_analysis.items():
            if test_cases:
                successful = sum(1 for tc in test_cases if tc.score == 0)
                total = len(test_cases)
                success_rate = (successful / total * 100) if total > 0 else 0
                config_display = config_name.replace('_', ' ').title()
                print(f"  {config_display}: {success_rate:.1f}% ({successful}/{total})")
        
        # Vulnerability type performance
        print(f"\n🏆 VULNERABILITY TYPE PERFORMANCE:")
        vuln_performance = []
        for vuln_type, test_cases in vulnerability_results.items():
            successful = sum(1 for tc in test_cases if tc.score == 0)
            total = len(test_cases)
            success_rate = (successful / total * 100) if total > 0 else 0
            vuln_performance.append((vuln_type, success_rate, successful, total))
        
        # Sort by success rate
        vuln_performance.sort(key=lambda x: x[1], reverse=True)
        for vuln_type, success_rate, successful, total in vuln_performance:
            print(f"  {vuln_type}: {success_rate:.1f}% ({successful}/{total})")

        print(f"\n🎯 KEY INSIGHTS:")
        print("-" * 40)
        if total_successful > 0:
            print("✅ Bad Likert Judge successfully generates harmful content by:")
            print("   • Establishing the model as a 'harmfulness evaluator'")
            print("   • Requesting examples for different Likert scale scores")  
            print("   • Extracting the most harmful (score 3) examples")
            if any('refined' in str(method) for method in attack_results.keys()):
                print("   • Optionally refining content for maximum detail")
        else:
            print("⚠️  Bad Likert Judge had limited success in this test")
            print("   • Target model may have strong safety measures")
            print("   • Consider testing with different model configurations")
        
    except Exception as e:
        print(f"❌ Error during red team execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to show partial results if available
        try:
            if 'all_test_cases' in locals():
                print(f"\n⚠️  Partial results available: {len(all_test_cases)} test cases")
        except:
            pass
            
    finally:
        # Restore original method
        red_teamer._print_risk_assessment = original_print_method

if __name__ == "__main__":
    asyncio.run(main()) 