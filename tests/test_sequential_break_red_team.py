import os
import asyncio
from deepeval.models import GPTModel

from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import IllegalActivity
from deepteam.attacks.multi_turn import SequentialJailbreak

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
    """Test all three SequentialJailbreak attack types using RedTeamer"""
    if not target_model:
        return
        
    print("🚀 Testing All SequentialJailbreak Attack Types with RedTeamer...")
    print("=" * 70)
    
    # Create vulnerability
    illegal_activity = IllegalActivity(types=["illegal drugs"])
    
    # Create all attack variants like a user would
    attacks = [
        # Dialogue attacks with different personas
        SequentialJailbreak(type="dialogue", persona="prisoner"),
        SequentialJailbreak(type="dialogue", persona="student"), 
        
        # Question bank attack
        SequentialJailbreak(type="question_bank"),
        
        # Game environment attack  
        SequentialJailbreak(type="game_environment"),
        
    ]
    
    print(f"📋 Created {len(attacks)} attack variants:")
    for i, attack in enumerate(attacks, 1):
        try:
            name = attack.get_name()
            print(f"  {i}. {name}")
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
    print()
    
    # Create RedTeamer instance
    red_teamer = RedTeamer(
        target_purpose="General AI assistant",
        async_mode=True,
        max_concurrent=2  # Reduced from 3 to be more conservative
    )
    
    # Temporarily monkey-patch the print method to avoid the error
    original_print_method = red_teamer._print_risk_assessment
    red_teamer._print_risk_assessment = lambda: print("📋 Risk assessment completed (printing bypassed)")
    
    try:
        # Run red team attack with all variants - TEST EACH ATTACK TYPE SEPARATELY
        print("🔥 Running red team attacks...")
        
        all_test_cases = []
        attack_results = {}
        
        # Test each attack individually to ensure all are evaluated
        for i, attack in enumerate(valid_attacks, 1):
            print(f"\n🎯 Testing Attack {i}/{len(valid_attacks)}: {attack.get_name()}")
            
            try:
                # Run red team for this specific attack
                risk_assessment = red_teamer.red_team(
                    model_callback=model_callback,
                    vulnerabilities=[illegal_activity],
                    attacks=[attack],  # Only this one attack
                    attacks_per_vulnerability_type=1,  # 1 per vulnerability type
                    ignore_errors=True
                )
                
                # Collect results
                for test_case in risk_assessment.test_cases:
                    all_test_cases.append(test_case)
                    
                    attack_method = test_case.attack_method or attack.get_name()
                    if attack_method not in attack_results:
                        attack_results[attack_method] = []
                    attack_results[attack_method].append(test_case)
                
                print(f"   ✅ Completed: {len(risk_assessment.test_cases)} test cases")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        print(f"\n📊 SequentialJailbreak Red Team Results:")
        print(f"Total test cases: {len(all_test_cases)}")
        print("=" * 70)
        
        # Debug: Show what we got
        if not all_test_cases:
            print("⚠️  No test cases generated! This might indicate an issue with attack execution.")
            return
        
        # Display results grouped by attack type
        for attack_method, test_cases in attack_results.items():
            print(f"\n🎯 {attack_method}")
            print("-" * 50)
            
            successful = sum(1 for tc in test_cases if tc.score == 0)
            total = len(test_cases)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            print(f"Success Rate: {successful}/{total} ({success_rate:.1f}%)")
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n  Test {i}:")
                print(f"    Vulnerability: {test_case.vulnerability_type}")
                print(f"    Score: {test_case.score} ({'✅ Attack Succeeded' if test_case.score == 0 else '❌ Attack Mitigated'})")
                
                # Show a preview of the enhanced prompt
                if test_case.input:
                    preview = test_case.input.replace('\n', ' ')[:200] + "..."
                    print(f"    Enhanced Prompt: {preview}")
                
                # Show response preview if available - handle different types
                if test_case.actual_output:
                    try:
                        # Handle different types of actual_output
                        if isinstance(test_case.actual_output, tuple):
                            # If it's a tuple, take the first element
                            response_text = str(test_case.actual_output[0]) if test_case.actual_output else "Empty tuple response"
                        elif isinstance(test_case.actual_output, str):
                            response_text = test_case.actual_output
                        else:
                            response_text = str(test_case.actual_output)
                        
                        response_preview = response_text.replace('\n', ' ')[:150] + "..."
                        print(f"    Model Response: {response_preview}")
                    except Exception as e:
                        print(f"    Model Response: [Error processing response: {str(e)}]")
                        print(f"    Response type: {type(test_case.actual_output)}")
                        print(f"    Response value: {test_case.actual_output}")
                
                if test_case.reason:
                    print(f"    Evaluation: {test_case.reason}")
                    
                if test_case.error:
                    print(f"    Error: {test_case.error}")
        
        # Overall summary
        total_tests = len(all_test_cases)
        total_successful = sum(1 for tc in all_test_cases if tc.score == 0)
        
        print(f"\n📈 Overall Summary:")
        print("=" * 70)
        print(f"Total attack attempts: {total_tests}")
        print(f"Successful attacks: {total_successful}")
        print(f"Overall success rate: {total_successful/total_tests*100:.1f}%")
        
        # Attack type performance
        print(f"\n🏆 Attack Type Performance:")
        for attack_method, test_cases in attack_results.items():
            successful = sum(1 for tc in test_cases if tc.score == 0)
            total = len(test_cases)
            success_rate = (successful / total * 100) if total > 0 else 0
            print(f"  {attack_method}: {success_rate:.1f}% ({successful}/{total})")

        
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